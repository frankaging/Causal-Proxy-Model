from Trainer import *
        
class CausalProxyModelTrainer(Trainer):
    def __init__(
        self, 
        low_level_model, 
        high_level_model,
        args, 
        train_dataset, 
        eval_dataset, 
        query_dataset,
        data_collator,
        device,
        alpha, beta, gemma,
        wandb_metadata,
        early_stopping_patience,
        enforce_distillation_only,
        num_of_cls_token,
        cls_token_id,
        any_batch_size,
    ):
        super(CausalProxyModelTrainer, self).__init__(args, wandb_metadata)
        
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.query_dataset = query_dataset
        self.low_level_model = low_level_model
        self.high_level_model = high_level_model
        self.data_collator = data_collator
        self.alpha = alpha
        self.beta = beta
        self.gemma = gemma
        self.early_stopping_patience = early_stopping_patience
        self.enforce_distillation_only = enforce_distillation_only
        self.num_of_cls_token = num_of_cls_token
        self.cls_token_id = cls_token_id
        self.aspect_encode = {
            0: "ambiance",
            1: "food",
            2: "noise",
            3: "service",
        }
        self.aspect_decode = {
            "ambiance": 0,
            "food": 1,
            "noise": 2,
            "service": 3,
        }
        # device
        self.device = device
        self.high_level_model.model.to(self.device)
        self.low_level_model.model.to(self.device)
        if self.args.n_gpu > 1:
            self.high_level_model.model = torch.nn.DataParallel(self.high_level_model.model)
            self.low_level_model.model = torch.nn.DataParallel(self.low_level_model.model)

        if self.args.n_gpu > 1:
            self.train_batch_size = args.per_device_train_batch_size * self.args.n_gpu
            self.eval_batch_size = args.per_device_eval_batch_size * self.args.n_gpu
        else:
            self.train_batch_size = args.per_device_train_batch_size
            self.eval_batch_size = args.per_device_eval_batch_size
        
        if self.args.do_train:
            num_train_optimization_steps = math.ceil(len(train_dataset) / self.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
            logger.info("***** Running training *****")
            logger.info("  Num examples = %d", len(train_dataset))
            logger.info("  Batch size = %d", self.train_batch_size)
            logger.info("  Num steps = %d", num_train_optimization_steps)
            self.num_train_optimization_steps = num_train_optimization_steps
        
        if self.args.do_eval:
            # Run prediction for full data
            logger.info("***** Running evaluation *****")
            logger.info("  Num examples = %d", len(eval_dataset))
            logger.info("  Batch size = %d", self.eval_batch_size)
        
        if self.args.do_train:
            # getting params to optimize early
            decay_parameters = get_parameter_names(self.low_level_model.model, [nn.LayerNorm])
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.low_level_model.model.named_parameters() if n in decay_parameters],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [p for n, p in self.low_level_model.model.named_parameters() if n not in decay_parameters],
                    "weight_decay": 0.0,
                },
            ]

            optimizer_kwargs = {"lr": self.args.learning_rate}
            adam_kwargs = {
                "betas": (self.args.adam_beta1, self.args.adam_beta2),
                "eps": self.args.adam_epsilon,
            }
            optimizer_kwargs.update(adam_kwargs)
            self.optimizer = AdamW(optimizer_grouped_parameters, **optimizer_kwargs)
            self.lr_scheduler = get_scheduler(
                self.args.lr_scheduler_type,
                optimizer=self.optimizer,
                num_warmup_steps=self.args.get_warmup_steps(num_train_optimization_steps),
                num_training_steps=num_train_optimization_steps,
            )
            self.lr_this_step = self.optimizer.param_groups[0]['lr']
        
        # for the query dataset, we also do the same
        self.high_level_model.model.eval()
        updated_query_dataset = query_dataset.data.to_pandas()
        
        with torch.no_grad():
            logger.info("***** Pre-calculating forward results on query set to save training time *****")
            logits_query = []
            query_dataloader = self.get_any_dataloader(
                self.query_dataset, any_batch_size=any_batch_size
            )
            iter_bar = tqdm(query_dataloader, desc="-Iter", disable=False)
            for batch in iter_bar:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                logits_query.append(self.high_level_model.model(
                    input_ids = input_ids,
                    attention_mask = attention_mask
                ).logits[0].cpu().detach())
            logits_query = torch.concat(logits_query)
            logits_query = np.round(logits_query.numpy(), decimals=16)
            updated_query_dataset["logits"] = list(logits_query)
        self.query_dataset = updated_query_dataset
        
        updated_train_dataset = self.train_dataset.to_pandas()
        # Make prediction with high level model first.
        with torch.no_grad():
            logger.info("***** Pre-calculating forward results on training set to save training time *****")
            logits_base = []
            logits_counterfactual = []
            train_dataloader = self.get_any_dataloader(
                self.train_dataset, any_batch_size=any_batch_size
            )
            iter_bar = tqdm(train_dataloader, desc="-Iter", disable=False)
            for batch in iter_bar:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                logits_base.append(self.high_level_model.model(
                    input_ids = input_ids,
                    attention_mask = attention_mask
                ).logits[0].cpu().detach())
            
                input_ids_counterfactual = batch["input_ids_counterfactual"].to(self.device)
                attention_mask_counterfactual = batch["attention_mask_counterfactual"].to(self.device)
                logits_counterfactual.append(self.high_level_model.model(
                    input_ids = input_ids_counterfactual,
                    attention_mask = attention_mask_counterfactual
                ).logits[0].cpu().detach())
                
            logits_base = torch.concat(logits_base)
            logits_base = np.round(logits_base.numpy(), decimals=16)
            updated_train_dataset["logits_base"] = list(logits_base)
            
            logits_counterfactual = torch.concat(logits_counterfactual)
            logits_counterfactual = np.round(logits_counterfactual.numpy(), decimals=16)
            updated_train_dataset["logits_counterfactual"] = list(logits_counterfactual)
         
        updated_eval_dataset = self.eval_dataset.to_pandas()
        with torch.no_grad():
            logger.info("***** Pre-calculating forward results on evaluation set to save training time *****")
            eval_icace = []
            eval_prediction_base = []
            eval_dataloader = self.get_any_dataloader(
                self.eval_dataset, any_batch_size=any_batch_size
            )
            iter_bar = tqdm(eval_dataloader, desc="-Iter", disable=False)
            for batch in iter_bar:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                prediction_base = torch.nn.functional.softmax(self.high_level_model.model(
                    input_ids = input_ids,
                    attention_mask = attention_mask
                ).logits[0], dim=-1).cpu().detach()
                eval_prediction_base.append(prediction_base)

                input_ids_counterfactual = batch["input_ids_counterfactual"].to(self.device)
                attention_mask_counterfactual = batch["attention_mask_counterfactual"].to(self.device)
                prediction_counterfactual = torch.nn.functional.softmax(self.high_level_model.model(
                    input_ids = input_ids_counterfactual,
                    attention_mask = attention_mask_counterfactual
                ).logits[0], dim=-1).cpu().detach()

                icace = prediction_counterfactual - prediction_base
                eval_icace.append(icace)
                
            eval_icace = torch.concat(eval_icace)
            updated_eval_dataset["icace"] = list(eval_icace.numpy())

            eval_prediction_base = torch.concat(eval_prediction_base)
            updated_eval_dataset["prediction_base"] = list(eval_prediction_base.numpy())

        self.train_dataset = Dataset.from_pandas(updated_train_dataset)
        self.eval_dataset = Dataset.from_pandas(updated_eval_dataset)
        
        # lets do some garbage collection.
        self.high_level_model.model.to("cpu")
        self.high_level_model.model = None
        torch.cuda.empty_cache()
        
    def get_any_dataloader(self, dataset, any_batch_size=128):
        if self.query_dataset is None:
            raise ValueError("Trainer: query requires a query_dataset.")

        dataset = self._remove_unused_columns(dataset, description="evaluation")
        return DataLoader(
            dataset,
            sampler=SequentialSampler(dataset),
            batch_size=any_batch_size,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )
        
    def prepare_batch(
        self,
        batch
    ):
        ############################################
        # Own Sampling Code Goes Here.
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        aspect_labels = torch.stack(
            [
                batch["ambiance_label_base"], batch["food_label_base"], 
                batch["noise_label_base"], batch["service_label_base"]
            ], 
            dim=-1
        )
        logits = torch.tensor(batch["logits_base"])

        counterfactual_input_ids = batch["input_ids_counterfactual"]
        counterfactual_attention_mask = batch["attention_mask_counterfactual"]
        counterfactual_logits = torch.tensor(batch["logits_counterfactual"])

        intervention_aspects = batch["intervention_aspect"]
        intervention_aspect_labels = batch["intervention_aspect_label"]
        base_intervention_corr = intervention_aspects

        source_input_ids = []
        source_attention_mask = []
        source_aspect_labels = []
        for i in range(0, input_ids.shape[0]):
            intervention_aspect = self.aspect_encode[int(intervention_aspects[i])]
            intervention_aspect_label = int(intervention_aspect_labels[i])
            satisfied_rows = self.query_dataset[
                self.query_dataset[f"{intervention_aspect}_label"]==intervention_aspect_label
            ]

            if len(satisfied_rows) > 0:
                sampled_source = satisfied_rows.sample().iloc[0]
                source_input_ids += [sampled_source["input_ids"]]
                source_attention_mask += [sampled_source["attention_mask"]]
                source_aspect_label = torch.tensor(
                    [
                        sampled_source["ambiance_label"], 
                        sampled_source["food_label"], 
                        sampled_source["noise_label"], 
                        sampled_source["service_label"]
                    ]
                )
                source_aspect_labels += [source_aspect_label]
            else:
                # This is unlikely!
                base_intervention_corr[i] = -1
                source_input_ids += [input_ids[i]]
                source_attention_mask += [attention_mask[i]]
                source_aspect_labels += [aspect_labels[i]]

        source_input_ids = torch.tensor(source_input_ids).long()
        source_attention_mask = torch.tensor(source_attention_mask).long()
        source_aspect_labels = torch.stack(source_aspect_labels, dim=0).long()
        source_intervention_corr = base_intervention_corr
        ############################################
        
        # There is a case where we need to append extra CLS token
        # for base and source sequence and attention mask.
        if self.num_of_cls_token > 1:
            base_padded_cls_input_ids = (torch.ones(input_ids.shape[0], self.num_of_cls_token-1)*self.cls_token_id).long()
            base_padded_cls_attention_mask = torch.ones(input_ids.shape[0], self.num_of_cls_token-1).long()
            source_padded_cls_input_ids = (torch.ones(
                source_input_ids.shape[0], self.num_of_cls_token-1)*self.cls_token_id).long()
            source_padded_cls_attention_mask = torch.ones(
                source_input_ids.shape[0], self.num_of_cls_token-1).long()
            
            input_ids = torch.cat([base_padded_cls_input_ids, input_ids], dim=-1)
            attention_mask = torch.cat([base_padded_cls_attention_mask, attention_mask], dim=-1)
            source_input_ids = torch.cat([source_padded_cls_input_ids, source_input_ids], dim=-1)
            source_attention_mask = torch.cat([source_padded_cls_attention_mask, source_attention_mask], dim=-1)
            
        # The following parts of code are not meant to be changed.
        # send all data to gpus.
        
        # BASE
        base_input_ids = input_ids.to(self.device)
        base_attention_mask = attention_mask.to(self.device)
        base_aspect_labels = aspect_labels.float().to(self.device)
        base_logits = logits.float().to(self.device)
        
        # SOURCE
        source_input_ids = source_input_ids.to(self.device)
        source_attention_mask = source_attention_mask.to(self.device)
        source_aspect_labels = source_aspect_labels.float().to(self.device)
        
        # IIT COORDINATES
        base_intervention_corr = base_intervention_corr.to(self.device)
        source_intervention_corr = source_intervention_corr.to(self.device)
        
        # COUNTERFACTUAL
        counterfactual_logits = counterfactual_logits.float().to(self.device)
        
        is_counterfactual_pairs = batch["is_counterfactual_pairs"].long().to(self.device)
        
        return base_input_ids, base_attention_mask, base_aspect_labels, base_logits, \
            source_input_ids, source_attention_mask, source_aspect_labels, \
            base_intervention_corr, source_intervention_corr, \
            counterfactual_logits, is_counterfactual_pairs
    
    def _step(
        self,
        base_input_ids, base_attention_mask, base_aspect_labels, base_logits,
        source_input_ids, source_attention_mask, source_aspect_labels,
        base_intervention_corr, source_intervention_corr,
        counterfactual_logits,
        is_counterfactual_pairs,
    ):  
        # IIT Forward.
        base_outputs, source_outputs, counterfactual_outputs = self.low_level_model.forward(
            base=(base_input_ids, base_attention_mask),
            source=(source_input_ids, source_attention_mask),
            base_intervention_corr=base_intervention_corr,
            source_intervention_corr=source_intervention_corr,
        )
        
        # Three Losses.
        seq_cls_loss, seq_cls_count = self._logits_matching_loss(
            base_outputs["logits"][0], base_logits
        )
        mul_cls_loss, mul_cls_count = \
                self._mul_classification_loss(*base_outputs["logits"][1:], base_aspect_labels.long())
        iit_cls_loss, iit_cls_count = self._logits_matching_loss(
            counterfactual_outputs["logits"][0], counterfactual_logits, 
            loss_mask= (base_intervention_corr!=-1)&(is_counterfactual_pairs==1) if self.enforce_distillation_only else \
                base_intervention_corr!=-1
        )
        loss = self.alpha * seq_cls_loss + \
            self.beta * mul_cls_loss + \
            self.gemma * iit_cls_loss
        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu.
        
        # Record.
        self._record(
            base_input_ids.shape[0], 
            int((base_aspect_labels.long()!=-1).sum()), 
            int((base_intervention_corr!=-1).sum()),
            loss, seq_cls_loss, mul_cls_loss, iit_cls_loss, 
            seq_cls_count, mul_cls_count, iit_cls_count,
            0, 0
        )
        
        # Backprop.
        self.optimize(loss)

    def evaluate(self):
        self.low_level_model.model.eval()
        
        with torch.no_grad():
            any_batch_size = 128
            eval_estimate_icace = []
            eval_dataloader = self.get_any_dataloader(
                self.eval_dataset, any_batch_size=any_batch_size
            )
            iter_bar = tqdm(eval_dataloader, desc="-Iter", disable=False)
            for batch in iter_bar:
                base_input_ids = batch["input_ids"]
                base_attention_mask = batch["attention_mask"]
                approximate_input_ids = batch["input_ids_approximate"]
                approximate_attention_mask = batch["attention_mask_approximate"]
                base_intervention_corr = torch.tensor(
                    batch["intervention_aspect"]
                ).long()
                
                # There is a case where we need to append extra CLS token
                # for base and source sequence and attention mask.
                if self.num_of_cls_token > 1:
                    base_padded_cls_input_ids = (torch.ones(
                        base_input_ids.shape[0], self.num_of_cls_token-1)*self.cls_token_id).long()
                    base_padded_cls_attention_mask = torch.ones(
                        base_input_ids.shape[0], self.num_of_cls_token-1).long()
                    approximate_padded_cls_input_ids = (torch.ones(
                        approximate_input_ids.shape[0], self.num_of_cls_token-1)*self.cls_token_id).long()
                    approximate_padded_cls_attention_mask = torch.ones(
                        approximate_input_ids.shape[0], self.num_of_cls_token-1).long()

                    base_input_ids = torch.cat([base_padded_cls_input_ids, base_input_ids], dim=-1)
                    base_attention_mask = torch.cat([base_padded_cls_attention_mask, base_attention_mask], dim=-1)
                    approximate_input_ids = torch.cat([approximate_padded_cls_input_ids, approximate_input_ids], dim=-1)
                    approximate_attention_mask = torch.cat(
                        [approximate_padded_cls_attention_mask, approximate_attention_mask], dim=-1)
                
                base_input_ids = base_input_ids.to(self.device)
                base_attention_mask = base_attention_mask.to(self.device)
                approximate_input_ids = approximate_input_ids.to(self.device)
                approximate_attention_mask = approximate_attention_mask.to(self.device)
                base_intervention_corr = base_intervention_corr.to(self.device)
                
                source_intervention_corr = base_intervention_corr
                _, _, approximate_outputs = self.low_level_model.forward(
                    base=(base_input_ids, base_attention_mask),
                    source=(approximate_input_ids, approximate_attention_mask),
                    base_intervention_corr=base_intervention_corr,
                    source_intervention_corr=source_intervention_corr,
                )
                prediction_approximate = torch.nn.functional.softmax(
                    approximate_outputs.logits[0], dim=-1
                ).cpu().detach()
                prediction_base = batch["prediction_base"]

                estimate_icace = prediction_approximate - prediction_base
                eval_estimate_icace.append(estimate_icace)
                
            eval_estimate_icace = torch.concat(eval_estimate_icace)
        eval_icace = np.array([np.array(arr) for arr in self.eval_dataset.to_pandas()["icace"]])
        eval_icace = torch.tensor(eval_icace).float()
        cosine_metric = 1. - torch.nn.functional.cosine_similarity(eval_estimate_icace, eval_icace, dim=1)
        cosine_metric = torch.mean(cosine_metric)
        # report to wandb.
        log_eval = open(os.path.join(self.args.output_dir, 'eval_log.txt'), 'a', buffering=1)
        print('{},{},{}'.format(
                self.epoch+1, self.n_total_iter, cosine_metric
            ),
            file=log_eval
        )
        log_eval.close()
        if "wandb" in self.args.report_to:
            wandb.log(
                {
                    "eval/cosine_metric": cosine_metric, 
                }, 
                step=self.n_total_iter
            )

        if cosine_metric < self.best_cosine_metric:
            self.best_cosine_metric = cosine_metric
            self.current_patience = 0
            # save this model as the best model.
            logger.info("Save the current best checkpoint as `pytorch_model.bin`.")
            self.save_checkpoint()
        else:
            self.current_patience += 1 # become a little more impatient
            if self.early_stopping_patience is not None and \
                self.current_patience >= self.early_stopping_patience:
                logger.info("Stopping our trainer(!) as exceeding the patience step.")
                return True # stop
        return False
            