import marimo

__generated_with = "0.18.1"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""
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
    with open("secrets/OPENAIKEY.txt", "r") as f:
        os.environ["OPENAI_API_KEY"] = f.read()
    client = OpenAI()
    return (client,)


@app.cell
def _(pl):
    df = pl.read_csv("data/mitweet_sample.csv")
    print(f"Loaded {df.height} tweets")
    return (df,)


@app.cell
def _(df):
    df.head(10)
    return


@app.cell
def _(mo):
    mo.md(r"""
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
        m = re.search(r"<output>\s*(.*?)\s*</output>", text, flags=re.DOTALL | re.IGNORECASE)
        if m:
            text = m.group(1).strip()
        

        text = text.strip().upper()
        valid_labels = ["LEFT", "RIGHT", "CENTER", "MIXED", "NONE"]
        for label in valid_labels:
            if label in text:
                return label
        
        return text 
    
    def _query_llm(row: dict) -> dict:
        """Query the LLM for a single tweet."""
        tweet = row["tweet"]
        prompt = IMPROVED_PROMPT_TEMPLATE.format(tweet=tweet)
        resp = client.responses.create(model="gpt-4.1-mini", input=prompt)
        output_text = getattr(resp, "output_text", "") or ""
        if "<output>" in output_text and "</output>" not in output_text:
            output_text = output_text + "</output>"
        return output_text
    
    start_time = time.time()
    results = []
    for row in tqdm(df.iter_rows(named=True), total=df.height, desc="Labeling tweets"):
        output_text = _query_llm(row)
        prediction = _parse_output(output_text)

        result_row = {**row, **{"llm_output": output_text, "prediction": prediction}}
        results.append(result_row)
    
    elapsed_time = time.time() - start_time
    
    improved_predictions = pl.DataFrame(results)
    
    print(f"\nTotal time: {elapsed_time:.2f} seconds")
    print(f"Average time per tweet: {elapsed_time/df.height:.2f} seconds")
    
    return (results, improved_predictions, elapsed_time)


@app.cell
def _(mo):
    mo.md(r"""
    """)
    return


@app.cell
def _(improved_predictions):
    improved_predictions
    return


@app.cell
def _(improved_predictions, pl):
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
    """)
    return


@app.cell
def _(improved_predictions, pl):
    correct = (improved_predictions["partisan_lean"] == improved_predictions["prediction"]).sum()
    total = improved_predictions.height
    accuracy = correct / total
    
    print(f"Overall Accuracy: {accuracy:.1%} ({correct}/{total})")
    
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

