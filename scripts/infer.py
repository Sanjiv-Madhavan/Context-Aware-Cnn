from unet_denoising.cli import main

if __name__ == "__main__":
    import sys

    sys.argv = [sys.argv[0], "--config", "configs/config.yaml", "infer"]
    main()
