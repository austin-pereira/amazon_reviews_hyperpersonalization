
import random
from typing import Tuple
import pandas as pd

def _sample_reviews() -> pd.DataFrame:
    # Small, hand‑curated Amazon‑style snippet for offline demo
    data = [
        # text, sentiment(0=neg,1=pos), persona ("eco","value","luxury")
        ("Loved the battery life and recycled packaging. Will buy again.", 1, "eco"),
        ("Too expensive for what it does. Not satisfied.", 0, "value"),
        ("Elegant feel, premium finish. Worth the price.", 1, "luxury"),
        ("Arrived late, customer service was unhelpful.", 0, "value"),
        ("Great sound quality, fair price. Highly recommend!", 1, "value"),
        ("Packaging was wasteful and product felt cheap.", 0, "eco"),
        ("Sleek design, perfect for gifting. Luxurious experience.", 1, "luxury"),
        ("Stopped working after a week. Disappointed.", 0, "value"),
        ("Sustainable materials, brand aligns with my values.", 1, "eco"),
        ("Average build, okay for the cost.", 0, "value"),
        ("Premium leather and craftsmanship, exceeded expectations.", 1, "luxury"),
        ("Poor instructions, had to return it.", 0, "value"),
        ("Eco‑friendly refills saved me money long term.", 1, "eco"),
        ("Shiny but flimsy, not worth it.", 0, "value"),
        ("Luxury feel, fast shipping, immaculate unboxing.", 1, "luxury"),
        ("Recyclable parts and low energy use are a plus.", 1, "eco"),
        ("Faulty unit received, annoyed with support.", 0, "value"),
        ("Affordable, reliable, does the job.", 1, "value"),
        ("Rich materials, top‑tier experience.", 1, "luxury"),
        ("Green packaging but product underperformed.", 0, "eco"),
    ]
    df = pd.DataFrame(data, columns=["text", "label_sentiment", "persona"])
    return df.sample(frac=1.0, random_state=42).reset_index(drop=True)

# Switch to Hugging Face if internet is available.
USE_HF = True

def load_datasets(test_size: float = 0.25, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if USE_HF:
        try:
            from datasets import load_dataset
            ds_train = load_dataset("amazon_polarity", split="train[:50%]")
            ds_test  = load_dataset("amazon_polarity", split="test[:10%]")
            import pandas as pd
            train_df = pd.DataFrame(ds_train)[["content", "label"]].rename(columns={"content":"text","label":"label_sentiment"})
            test_df  = pd.DataFrame(ds_test)[["content", "label"]].rename(columns={"content":"text","label":"label_sentiment"})
            # No persona labels in amazon_polarity; create weak personas via keywords (for demo only).
            def infer_persona(t: str) -> str:
                t_low = t.lower()
                if any(k in t_low for k in ["eco", "green", "sustain", "recycl"]):
                    return "eco"
                if any(k in t_low for k in ["luxury", "premium", "leather", "elegant"]):
                    return "luxury"
                return "value"
            train_df["persona"] = train_df["text"].apply(infer_persona)
            test_df["persona"]  = test_df["text"].apply(infer_persona)
            return train_df, test_df
        except Exception as e:
            print("Falling back to sample dataset due to:", e)

    df = _sample_reviews()
    # simple split
    import math
    n_test = max(1, int(len(df) * test_size))
    test_df = df.iloc[:n_test].reset_index(drop=True)
    train_df = df.iloc[n_test:].reset_index(drop=True)
    return train_df, test_df
