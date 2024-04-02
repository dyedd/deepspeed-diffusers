def deepspeed_config_from_args(args):
    return {
        # train_batch_size = train_micro_batch_size_per_gpu * gradient_accumulation *GPU
        'train_micro_batch_size_per_gpu': args.train_micro_batch_size_per_gpu,
        'gradient_accumulation_steps': args.gradient_accumulation_steps,
        'fp16': args.use_fp16,
        "steps_per_print": args.log_interval,
        "communication_data_type": args.communication_data_type,
        "optimizer": args.optimizer,
        "scheduler": args.scheduler,
        "wall_clock_breakdown": args.wall_clock_breakdown,
        "wandb": args.wandb
    }
