from unet_denoising.cli import main

if __name__ == "__main__":
    import sys

    sys.argv = [sys.argv[0], "--config", "configs/config.yaml", "s3-pull"]
    main()
