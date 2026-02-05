import marimo

__generated_with = "0.19.7"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""
    # Setup
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

    return OpenAI, mo, os, pl, re, tqdm


@app.cell
def _(OpenAI, os):
    # Set your API key
    # Make sure to create the file OPENAIKEY.txt before running this
    # (You can use the OPENAIKEY.txt.template file as a template)
    with open("secrets/OPENAIKEY.txt", "r") as f:
        os.environ["OPENAI_API_KEY"] = f.read().strip()
    client = OpenAI()
    return (client,)


@app.cell
def _(pl):
    # Load MITweet dataset
    df = pl.read_csv("data/mitweet_sample.csv")
    return (df,)


@app.cell
def _(df):
    # Look at data
    df
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Labeling
    """)
    return


@app.cell
def _():
    SIMPLE_PROMPT_TEMPLATE_change = """You are helping label political tweets by their partisan lean.
    {tweet}

    # ANALYSIS INSTRUCTIONS

    You are helping label political tweets by their partisan leaning.
    Read the tweet and consider the following steps:

    1. What is the tweet primarily expressing, arguing, or responding to?
    2. Does the tweet explicitly support, criticize, or question any political ideology,
    political group, policy, or public figure?
    3. Is there any relevant political or social context that helps explain this tweet,
    such as current events, political debates, or commonly discussed issues?
    4. Based on the tweet and its context, determine which political stance 
    (left-wing, centrist, or right-wing) the tweet best aligns with.

    If you believe the tweet does not express any political opinion or stance,
    please mark it as "None."

    # RESPONSE FORMAT

    <analysis>
    **Main idea:**  
    Briefly explain what the tweet is mainly saying or arguing (1–2 sentences).

    **Context:** 
    Describe any relevant political or social background that helps explain the tweet[1-2 sentence]

    **Directional assessment:** [Direction] because [1-2 sentence reason]
    </analysis>
    <output>
    [LEFT/CENTER/RIGHT/MIXED]
    </output>

    Or if tweet is not political:
    <output>
    NONE
    </output>
    """
    return (SIMPLE_PROMPT_TEMPLATE_change,)


@app.cell
def _(SIMPLE_PROMPT_TEMPLATE_change, client, df, pl, re, tqdm):
    def _parse_output(output_text: str) -> str:
        text = (output_text or "").strip()
        # Prefer the explicit <output> block if present
        m = re.search(
            r"<output>\s*(.*?)\s*</output>", text, flags=re.DOTALL | re.IGNORECASE
        )
        if m:
            text = m.group(1).strip()
        return text.strip().upper()

    def _query_llm(row: dict) -> dict:
        tweet = row["tweet"]
        prompt = SIMPLE_PROMPT_TEMPLATE_change.format(tweet=tweet)
        resp = client.responses.create(model="gpt-4.1-mini", input=prompt, max_output_tokens=160)
        output_text = getattr(resp, "output_text", "") or ""
        return output_text

    # Process rows with a for-loop
    results = []
    for row in tqdm(df.iter_rows(named=True), total=df.height):
        output_text = _query_llm(row)
        prediction = _parse_output(output_text)
        # Combine original row data with classification results
        result_row = {**row, **{"llm_output": output_text, "prediction": prediction}}
        results.append(result_row)

    # Convert results back to a DataFrame
    simple_predictions = pl.DataFrame(results)
    return (results,)


@app.cell
def _(pl, results):
    pl.DataFrame(results)
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Evaluation
    """)
    return


@app.cell
def _(pl, results):
    # Look at results
    pl.DataFrame(results)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Confusion Matrix
    """)
    return


@app.cell
def _(pl, results):
    crosstab = (
        pl.DataFrame(results)
        .group_by("partisan_lean", "prediction")
        .len()
        .pivot(index="partisan_lean", on="prediction", values="len")
    )

    # Get prediction columns (everything except the index)
    prediction_columns = [col for col in crosstab.columns if col != "partisan_lean"]

    crosstab = crosstab.with_columns(
        pl.concat_str([pl.lit("actually_"), pl.col("partisan_lean")]).alias(
            "partisan_lean"
        )
    ).rename(
        {
            "partisan_lean": "actual_label",
            **{col: f"predicted_{col}" for col in prediction_columns},
        }
    )

    crosstab
    return


if __name__ == "__main__":
    app.run()
