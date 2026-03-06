import argparse
import subprocess


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 999])
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--python_bin", type=str, default="python")
    args = p.parse_args()

    variants = [
        ("full_rjp", []),
        ("no_ram", ["--disable_ram"]),
        ("no_counterfactual", ["--disable_counterfactual"]),
        ("no_anatomy", ["--disable_anatomy"]),
        ("fixed_fusion", ["--disable_adaptive_fusion", "--fixed_fusion_alpha", "0.5"]),
    ]

    for seed in args.seeds:
        for name, extra in variants:
            run_name = f"{name}_seed{seed}"
            cmd = [
                args.python_bin,
                "run_paper1.py",
                "--data_root",
                args.data_root,
                "--run_name",
                run_name,
                "--epochs",
                str(args.epochs),
                "--batch_size",
                str(args.batch_size),
                "--seed",
                str(seed),
            ] + extra
            print("Running:", " ".join(cmd))
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
