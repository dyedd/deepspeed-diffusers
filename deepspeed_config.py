def deepspeed_config_from_args(args):
    return {
        # train_batch_size = train_micro_batch_size_per_gpu * gradient_accumulation *GPU
        'train_micro_batch_size_per_gpu': args.train_micro_batch_size_per_gpu,
        'gradient_accumulation_steps': args.gradient_accumulation_steps,
        'fp16': {
            'enabled': args.use_fp16,
            "auto_cast": False,
            "loss_scale": 0,
            "initial_scale_power": 16,
            "loss_scale_window": 1000,
            "hysteresis": 2,
            "consecutive_hysteresis": False,
            "min_loss_scale": 1
        },
        "activation_checkpointing": {
            "partition_activations": True,
            "cpu_checkpointing": True,
            "contiguous_memory_optimization": False,
            "number_checkpoints": None,
            "synchronize_checkpoint_boundary": False,
            "profile": True
        },
        "tensorboard": {
            "enabled": True,
            "output_path": f"tensorboard_logs",
            "job_name": f"{args.wandb_project}",
        },
        "steps_per_print": args.log_interval,
        "communication_data_type": args.communication_data_type,
        "optimizer": args.optimizer,
        "scheduler": args.scheduler,
        "wall_clock_breakdown": args.wall_clock_breakdown
    }
