import marimo

__generated_with = "0.18.1"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""
    # Political Ideology Classifier
    """)
    return


@app.cell
def _():
    import marimo as mo
    import re 
    import os
    import polars as pl
    from openai import OpenAI
    from tqdm import tqdm
    import time
    return OpenAI, mo, os, pl, re, tqdm, time


@app.cell
def _(OpenAI, os):
    # Set your API key
    with open("secrets/OPENAIKEY.txt", "r") as f:
        os.environ["OPENAI_API_KEY"] = f.read()
    client = OpenAI()
    return (client,)


@app.cell
def _(pl):
    # Load MITweet dataset
    df = pl.read_csv("data/mitweet_sample.csv")
    print(f"Loaded {df.height} tweets")
    return (df,)


@app.cell
def _(df):
    # Look at data structure
    df.head(10)
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Labeling
    """)
    return


@app.cell
def _():
    IMPROVED_PROMPT_TEMPLATE = """Classify the tweet's political ideology. Respond with ONLY the output tag, nothing else.

LEFT = Progressive/liberal (abortion rights, BLM, LGBTQ+, social justice) OR criticizes Republicans/Trump
RIGHT = Conservative (anti-abortion, border security, traditional values) OR criticizes Democrats/Biden
CENTER = Neutral analysis or reports on both sides
MIXED = Combines left and right positions
NONE = Not political

Examples:
"#BlackLivesMatter ride to #Ferguson has left me in awe." → <output>LEFT</output>
"Biden reached a deal with Mitch McConnell to support an anti-abortion judge." → <output>RIGHT</output>
"This was a calculated dog-whistle to what Democrats believe are moderate Republicans." → <output>CENTER</output>
"Hey GOP Senators. Since you voted against abortion rights, how about being pro-life for seniors?" → <output>MIXED</output>
"Great workout today!" → <output>NONE</output>

Tweet: {tweet}
<output>"""

    return (IMPROVED_PROMPT_TEMPLATE,)


@app.cell
def _(IMPROVED_PROMPT_TEMPLATE, client, df, pl, re, tqdm, time):
    def _parse_output(output_text: str) -> str:
        """Parse the classification from the LLM output."""
        text = (output_text or "").strip()
        # Prefer the explicit <output> block if present
        m = re.search(r"<output>\s*(.*?)\s*</output>", text, flags=re.DOTALL | re.IGNORECASE)
        if m:
            text = m.group(1).strip()
        
        # Also check for standalone labels
        text = text.strip().upper()
        valid_labels = ["LEFT", "RIGHT", "CENTER", "MIXED", "NONE"]
        for label in valid_labels:
            if label in text:
                return label
        
        return text  # Return as-is if no match
    
    def _query_llm(row: dict) -> dict:
        """Query the LLM for a single tweet."""
        tweet = row["tweet"]
        prompt = IMPROVED_PROMPT_TEMPLATE.format(tweet=tweet)
        resp = client.responses.create(model="gpt-4.1-mini", input=prompt)
        output_text = getattr(resp, "output_text", "") or ""
        # Ensure closing tag if missing
        if "<output>" in output_text and "</output>" not in output_text:
            output_text = output_text + "</output>"
        return output_text
    
    # Process rows with timing
    start_time = time.time()
    results = []
    for row in tqdm(df.iter_rows(named=True), total=df.height, desc="Labeling tweets"):
        output_text = _query_llm(row)
        prediction = _parse_output(output_text)
        # Combine original row data with classification results
        result_row = {**row, **{"llm_output": output_text, "prediction": prediction}}
        results.append(result_row)
    
    elapsed_time = time.time() - start_time
    
    # Convert results back to a DataFrame
    improved_predictions = pl.DataFrame(results)
    
    print(f"\nTotal time: {elapsed_time:.2f} seconds")
    print(f"Average time per tweet: {elapsed_time/df.height:.2f} seconds")
    
    return (results, improved_predictions, elapsed_time)


@app.cell
def _(mo):
    mo.md(r"""
    # Evaluation
    """)
    return


@app.cell
def _(improved_predictions):
    # Look at results
    improved_predictions
    return


@app.cell
def _(improved_predictions, pl):
    # Diagnostic: Check some outputs to see what the model is producing
    print("Sample outputs (first 5 rows):")
    sample = improved_predictions.head(5).select(["tweet", "partisan_lean", "prediction", "llm_output"])
    for sample_row in sample.iter_rows(named=True):
        print(f"\nActual: {sample_row['partisan_lean']} | Predicted: {sample_row['prediction']}")
        print(f"Tweet: {sample_row['tweet'][:100]}...")
        print(f"LLM output: {sample_row['llm_output'][:200]}...")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Accuracy Metrics
    """)
    return


@app.cell
def _(improved_predictions, pl):
    # Calculate accuracy
    correct = (improved_predictions["partisan_lean"] == improved_predictions["prediction"]).sum()
    total = improved_predictions.height
    accuracy = correct / total
    
    print(f"Overall Accuracy: {accuracy:.1%} ({correct}/{total})")
    
    # Per-category accuracy
    category_accuracy = (
        improved_predictions
        .with_columns(
            (pl.col("partisan_lean") == pl.col("prediction")).alias("is_correct")
        )
        .group_by("partisan_lean")
        .agg([
            pl.col("is_correct").sum().alias("correct"),
            pl.col("is_correct").count().alias("total"),
            (pl.col("is_correct").sum() / pl.col("is_correct").count()).alias("accuracy")
        ])
        .sort("partisan_lean")
    )
    
    print("\nPer-Category Accuracy:")
    category_accuracy
    return (accuracy, category_accuracy, correct, total)


@app.cell
def _(mo):
    mo.md(r"""
    ## Confusion Matrix
    """)
    return


@app.cell
def _(improved_predictions, pl):
    crosstab = (
        improved_predictions
        .group_by('partisan_lean', 'prediction')
        .len()
        .pivot(index="partisan_lean", on="prediction", values="len")
    )

    # Get prediction columns (everything except the index)
    prediction_columns = [col for col in crosstab.columns if col != "partisan_lean"]

    crosstab = (
        crosstab
        .with_columns(
            pl.concat_str([pl.lit("actual_"), pl.col("partisan_lean")]).alias("partisan_lean")
        )
        .rename({
            "partisan_lean": "actual_label",
            **{col: f"predicted_{col}" for col in prediction_columns}
        })
    )

    crosstab
    return


@app.cell
def _(improved_predictions, elapsed_time, accuracy, total):
    print(f"\nSummary:")
    print(f"Total tweets: {total}")
    print(f"Overall accuracy: {accuracy:.1%}")
    print(f"Total time: {elapsed_time:.2f} seconds")
    print(f"Average time per tweet: {elapsed_time/total:.3f} seconds")
    return (accuracy, elapsed_time, total)


if __name__ == "__main__":
    app.run()

