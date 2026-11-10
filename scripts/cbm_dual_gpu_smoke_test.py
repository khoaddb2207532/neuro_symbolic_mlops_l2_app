"""Run a lightweight data/model smoke test for the dual-GPU CBM notebook."""

import argparse

import torch

from src.data.concepts import ConceptAnnotatedDataset, load_concept_annotations
from src.data.dataset import create_dataloaders
from src.models.cbm import ConceptBottleneckModel
from src.models.cnn import ImageClassificationBaseline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--concept-annotations-path", required=True)
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--num-classes", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--num-workers", type=int, required=True)
    parser.add_argument("--seed-a", type=int, required=True)
    parser.add_argument("--seed-b", type=int, required=True)
    args = parser.parse_args()

    if torch.cuda.device_count() < 2:
        raise RuntimeError("Dual-GPU smoke test requires at least two CUDA GPUs.")

    print(">>> Running data and model smoke test on both GPUs...")
    _, train_loader, _, _ = create_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed_a,
    )

    images, labels = next(iter(train_loader))
    print(f"  [1/4] Training batch: images={tuple(images.shape)}, labels={tuple(labels.shape)}")
    assert images.shape[0] == args.batch_size, "Unexpected batch size."

    annotations, names, n_concepts, class_fallback = load_concept_annotations(
        args.concept_annotations_path
    )
    concept_dataset = ConceptAnnotatedDataset(
        train_loader.dataset,
        annotations,
        class_fallback,
        n_concepts=n_concepts,
    )
    concept_image, concept_label, concept_vector = concept_dataset[0]
    print(
        "  [2/4] Concept sample: "
        f"image={tuple(concept_image.shape)}, label={concept_label}, "
        f"concept_vector={tuple(concept_vector.shape)}"
    )

    baseline = ImageClassificationBaseline(
        architecture=args.backbone,
        num_classes=args.num_classes,
        pretrained=False,
    ).to("cuda:0")
    with torch.no_grad():
        baseline_output = baseline(images.to("cuda:0"))
    print(
        f"  [3/4] GPU 0 (seed {args.seed_a}): "
        f"logits={tuple(baseline_output[0].shape)} | "
        f"VRAM={torch.cuda.memory_allocated(0) / (1024**2):.1f} MB"
    )

    cbm = ConceptBottleneckModel(
        architecture=args.backbone,
        n_concepts=n_concepts,
        num_classes=args.num_classes,
        pretrained=False,
        concept_names=names,
    ).to("cuda:1")
    with torch.no_grad():
        cbm_output = cbm(images.to("cuda:1"))
    print(
        f"  [4/4] GPU 1 (seed {args.seed_b}): "
        f"logits={tuple(cbm_output[0].shape)}, "
        f"concepts={tuple(cbm_output[1].shape)} | "
        f"VRAM={torch.cuda.memory_allocated(1) / (1024**2):.1f} MB"
    )

    del baseline, cbm, images, labels, baseline_output, cbm_output
    torch.cuda.empty_cache()
    print(">>> Smoke test passed; both GPUs and data pipeline are ready.")


if __name__ == "__main__":
    main()
