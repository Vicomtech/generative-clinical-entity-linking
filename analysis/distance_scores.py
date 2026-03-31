import os
import glob
import numpy as np
import pandas as pd
import torch
import faiss
from transformers import AutoTokenizer, AutoModel


def load_sapbert(device):
    """Load SapBERT model and tokenizer."""
    model_name = "cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR-large"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    return tokenizer, model


def encode_texts(texts, tokenizer, model, device, batch_size=64):
    """Encode a list of texts into embeddings using SapBERT (CLS pooling)."""
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        encoded = tokenizer(
            batch, padding=True, truncation=True, max_length=128, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            outputs = model(**encoded)
        # CLS token embedding
        cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        all_embeddings.append(cls_embeddings)
    return np.vstack(all_embeddings).astype("float32")


def compute_l2_distances(pred_embeddings, desc_embeddings):
    """Compute pairwise L2 (Euclidean) distance using FAISS for each row."""
    dim = desc_embeddings.shape[1]
    distances = np.empty(len(pred_embeddings), dtype="float32")
    # Use a flat L2 index: add each description vector individually
    # and search with the corresponding prediction vector
    for i in range(len(pred_embeddings)):
        index = faiss.IndexFlatL2(dim)
        index.add(desc_embeddings[i : i + 1])
        d, _ = index.search(pred_embeddings[i : i + 1], 1)
        distances[i] = d[0, 0]  # squared L2 distance from FAISS
    return distances


def process_folder(folder_path, tokenizer, model, device):
    """Process all TSV files in a folder, adding an l2_distance column."""
    tsv_files = sorted(glob.glob(os.path.join(folder_path, "*.tsv")))
    print(f"\nProcessing folder: {folder_path}  ({len(tsv_files)} files)")

    for fpath in tsv_files:
        fname = os.path.basename(fpath)
        df = pd.read_csv(fpath, sep="\t")

        if "pred" not in df.columns or "descripcion" not in df.columns:
            print(f"  SKIP {fname}: missing 'pred' or 'descripcion' column")
            continue

        # Drop rows where either column is NaN, keep track of indices
        mask = df["pred"].notna() & df["descripcion"].notna()
        valid_df = df.loc[mask]

        if len(valid_df) == 0:
            print(f"  SKIP {fname}: no valid rows")
            continue

        preds = valid_df["pred"].astype(str).tolist()
        descs = valid_df["descripcion"].astype(str).tolist()

        pred_emb = encode_texts(preds, tokenizer, model, device)
        desc_emb = encode_texts(descs, tokenizer, model, device)

        dists = compute_l2_distances(pred_emb, desc_emb)

        df["l2_distance"] = np.nan
        df.loc[mask, "l2_distance"] = dists

        df.to_csv(fpath, sep="\t", index=False)
        mean_dist = np.nanmean(dists)
        print(f"  {fname}: {len(valid_df)} rows, mean L2 = {mean_dist:.4f}")

    return tsv_files


def score_stats(scores, label):
    """Compute descriptive statistics for a set of scores."""
    row = {"group": label, "count": len(scores)}
    if len(scores) > 0:
        row.update(
            {
                "mean": round(float(np.mean(scores)), 4),
                "std": round(float(np.std(scores)), 4),
                "min": round(float(np.min(scores)), 4),
                "max": round(float(np.max(scores)), 4),
                "median": round(float(np.median(scores)), 4),
            }
        )
        for p in [5, 10, 25, 50, 75, 90, 95]:
            row[f"p{p}"] = round(float(np.percentile(scores, p)), 4)
    return row


def analyse_correct_vs_incorrect(folder_path):
    """For each TSV, split l2_distance by is_recall_correct and report stats."""
    tsv_files = sorted(glob.glob(os.path.join(folder_path, "*.tsv")))
    all_stats = []

    for fpath in tsv_files:
        fname = os.path.basename(fpath)
        df = pd.read_csv(fpath, sep="\t")

        if "l2_distance" not in df.columns or "is_recall_correct" not in df.columns:
            continue

        result_df = df.dropna(subset=["l2_distance"])
        if len(result_df) == 0:
            continue

        correct_mask = result_df["is_recall_correct"] == 1
        all_scores = result_df["l2_distance"].values
        correct_scores = result_df.loc[correct_mask, "l2_distance"].values
        incorrect_scores = result_df.loc[~correct_mask, "l2_distance"].values

        for scores, label in [
            (all_scores, "all"),
            (correct_scores, "correct"),
            (incorrect_scores, "incorrect"),
        ]:
            row = score_stats(scores, label)
            row["file"] = fname
            all_stats.append(row)

        print(
            f"  {fname}: correct={len(correct_scores)}, "
            f"incorrect={len(incorrect_scores)}, "
            f"mean_correct={np.mean(correct_scores):.4f}, "
            f"mean_incorrect={np.mean(incorrect_scores):.4f}"
            if len(incorrect_scores) > 0
            else f"  {fname}: correct={len(correct_scores)}, incorrect=0"
        )

    if all_stats:
        stats_df = pd.DataFrame(all_stats)
        # Reorder columns so file and group come first
        cols = ["file", "group"] + [c for c in stats_df.columns if c not in ("file", "group")]
        stats_df = stats_df[cols]
        out_path = os.path.join(folder_path, "_l2_distance_stats.tsv")
        stats_df.to_csv(out_path, sep="\t", index=False)
        print(f"  Stats saved to {out_path}")

    return all_stats


def extract_model_name(filename):
    """Derive a clean model name from a TSV filename."""
    name = filename.replace(".tsv", "")
    # Strip _not_accurate suffix
    name = name.replace("_not_accurate", "")
    # Strip _checkpoint-XXXXX suffix
    import re
    name = re.sub(r"_checkpoint-\d+", "", name)
    # Strip output_codiesp_ prefix
    name = re.sub(r"^output_codiesp_", "", name)
    return name


def build_summary_table(folders, base_dir):
    """Build a final per-model summary DataFrame across all folders."""
    rows = []

    for folder_path in folders:
        if not os.path.isdir(folder_path):
            continue
        folder_name = os.path.basename(folder_path)
        tsv_files = sorted(glob.glob(os.path.join(folder_path, "*.tsv")))

        for fpath in tsv_files:
            fname = os.path.basename(fpath)
            if fname.startswith("_"):  # skip our own stats files
                continue

            df = pd.read_csv(fpath, sep="\t")
            if "l2_distance" not in df.columns or "is_recall_correct" not in df.columns:
                continue

            result_df = df.dropna(subset=["l2_distance"])
            if len(result_df) == 0:
                continue

            model = extract_model_name(fname)
            correct_mask = result_df["is_recall_correct"] == 1
            all_dists = result_df["l2_distance"].values
            correct_dists = result_df.loc[correct_mask, "l2_distance"].values
            incorrect_dists = result_df.loc[~correct_mask, "l2_distance"].values

            row = {
                "model": model,
                "folder": folder_name,
                "n_total": len(all_dists),
                "n_correct": len(correct_dists),
                "n_incorrect": len(incorrect_dists),
                "mean_l2_all": round(float(np.mean(all_dists)), 4),
                "std_l2_all": round(float(np.std(all_dists)), 4),
                "median_l2_all": round(float(np.median(all_dists)), 4),
            }
            if len(correct_dists) > 0:
                row["mean_l2_correct"] = round(float(np.mean(correct_dists)), 4)
                row["std_l2_correct"] = round(float(np.std(correct_dists)), 4)
                row["median_l2_correct"] = round(float(np.median(correct_dists)), 4)
            if len(incorrect_dists) > 0:
                row["mean_l2_incorrect"] = round(float(np.mean(incorrect_dists)), 4)
                row["std_l2_incorrect"] = round(float(np.std(incorrect_dists)), 4)
                row["median_l2_incorrect"] = round(float(np.median(incorrect_dists)), 4)

            rows.append(row)

    summary_df = pd.DataFrame(rows)
    # Sort by folder then model
    summary_df = summary_df.sort_values(["folder", "model"]).reset_index(drop=True)

    out_path = os.path.join(base_dir, "l2_distance_summary_per_model.tsv")
    summary_df.to_csv(out_path, sep="\t", index=False)
    print(f"\nFinal summary saved to {out_path}")
    print(summary_df.to_string(index=False))
    return summary_df


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer, model = load_sapbert(device)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    folders = [
        os.path.join(base_dir, "error_analysis"),
    ]

    for folder in folders:
        if os.path.isdir(folder):
            process_folder(folder, tokenizer, model, device)
        else:
            print(f"Folder not found: {folder}")

    # --- L2 distance analysis: correct vs incorrect predictions ---
    print("\n=== L2 distance analysis (correct vs incorrect) ===")
    for folder in folders:
        if os.path.isdir(folder):
            analyse_correct_vs_incorrect(folder)

    # --- Final summary table per model ---
    print("\n=== Final summary per model ===")
    build_summary_table(folders, base_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()


