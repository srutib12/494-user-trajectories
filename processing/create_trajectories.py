from hashlib import md5

import polars as pl
from loguru import logger

logger.add("logs/create_trajectories.log", rotation="10 MB", level="DEBUG", serialize=True)

def enrich_with_intercepts_and_factors(scored_notes: pl.DataFrame) -> pl.DataFrame:
    I_AND_F_COLUMNS = {
        "CoreModel (v1.1)": ("coreNoteIntercept", "coreNoteFactor1"),
        "ExpansionModel (v1.1)": ("expansionNoteIntercept", "expansionNoteFactor1"),
        "ExpansionPlusModel (v1.1)": ("expansionPlusNoteIntercept", "expansionPlusNoteFactor1"),
        "GroupModel01 (v1.1)": ("groupNoteIntercept", "groupNoteFactor1"),
        "GroupModel02 (v1.1)": ("groupNoteIntercept", "groupNoteFactor1"),
        "GroupModel03 (v1.1)": ("groupNoteIntercept", "groupNoteFactor1"),
        "GroupModel04 (v1.1)": ("groupNoteIntercept", "groupNoteFactor1"),
        "GroupModel05 (v1.1)": ("groupNoteIntercept", "groupNoteFactor1"),
        "GroupModel06 (v1.1)": ("groupNoteIntercept", "groupNoteFactor1"),
        "GroupModel07 (v1.1)": ("groupNoteIntercept", "groupNoteFactor1"),
        "GroupModel08 (v1.1)": ("groupNoteIntercept", "groupNoteFactor1"),
        "GroupModel09 (v1.1)": ("groupNoteIntercept", "groupNoteFactor1"),
        "GroupModel10 (v1.1)": ("groupNoteIntercept", "groupNoteFactor1"),
        "GroupModel11 (v1.1)": ("groupNoteIntercept", "groupNoteFactor1"),
        "GroupModel12 (v1.1)": ("groupNoteIntercept", "groupNoteFactor1"),
        "GroupModel13 (v1.1)": ("groupNoteIntercept", "groupNoteFactor1"),
        "GroupModel14 (v1.1)": ("groupNoteIntercept", "groupNoteFactor1"),
        "MultiGroupModel01 (v1.0)": ("multiGroupNoteIntercept", "multiGroupNoteFactor1"),
        "TopicModel01 (v1.0)": ("topicNoteIntercept", "topicNoteFactor1"),
        "TopicModel02 (v1.0)": ("topicNoteIntercept", "topicNoteFactor1"),
        "TopicModel03 (v1.0)": ("topicNoteIntercept", "topicNoteFactor1"),
        "ScoringDriftGuard (v1.0)": (None, None),
        "NmrDueToMinStableCrhTime (v1.0)": (None, None),
        "InsufficientExplanation (v1.0)": (None, None),
    }

    scored_notes = (
        scored_notes
        .with_columns(scoreCreatedAtDt=pl.from_epoch(pl.col("createdAtMillis"), time_unit="ms"))
        # Infer the pre drift guard model
        .with_columns(
            preDriftModel=pl.when(pl.col("decidedBy").str.contains("ScoringDriftGuard"))
                            .then(pl.col("metaScorerActiveRules").str.split(",").list[-2])
                            .otherwise(pl.col("decidedBy"))
        )
        # Retrieve the intercept from the inferred model
        .with_columns(
            noteFinalIntercept=pl.coalesce([
                pl.when(pl.col("preDriftModel").str.starts_with(prefix))
                  .then(pl.col(intercept_col))
                for prefix, (intercept_col, _) in I_AND_F_COLUMNS.items()
                if intercept_col is not None
            ]),
            # Retrieve the factor from the inferred model
            noteFinalFactor=pl.coalesce([
                pl.when(pl.col("preDriftModel").str.starts_with(prefix))
                  .then(pl.col(factor_col))
                for prefix, (_, factor_col) in I_AND_F_COLUMNS.items()
                if factor_col is not None
            ]),
        )
        .rename({"finalRatingStatus": "noteFinalRatingStatus"})
        .select("noteId", "noteFinalRatingStatus", "numRatings", "decidedBy", "noteFinalIntercept", "noteFinalFactor")
    )
    return scored_notes

# Calculate calendar-based user month (months since first action) and calendar month
def _enrich_with_user_and_calendar_month(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        _actionDt=pl.from_epoch(pl.col("createdAtMillis"), time_unit="ms"),
        _firstActionDt=pl.from_epoch(pl.col("participantFirstActionMillis"), time_unit="ms"),
    ).with_columns(
        userMonth=(
            (pl.col("_actionDt").dt.year() - pl.col("_firstActionDt").dt.year()) * 12
            + pl.col("_actionDt").dt.month() - pl.col("_firstActionDt").dt.month()
        ).cast(pl.Int32),
        calendarMonth=pl.col("_actionDt").dt.strftime("%Y-%m"),
    ).drop("_actionDt", "_firstActionDt")

if __name__ == "__main__":
    # Load data 
    users       = pl.read_parquet("data/2026-02-03/userEnrollment.parquet") 
    notes       = pl.read_parquet("data/2026-02-03/notes.parquet")
    ratings     = pl.read_parquet("data/2026-02-03/noteRatings.parquet")
    requests    = pl.read_parquet("data/2026-01-09/noteRequests.parquet").rename({"userId": "requesterParticipantId"}) # Using the user-level requests, not post-level requests!
    statuses    = pl.read_csv    ("data/2026-02-27-note_status_records.csv")    # Processed statutes, taken from a scm-prep run on 2/27.
    partisanship= pl.read_csv("data/renault_partisanship_labels.csv") # Partisanship data is from paper: "Republicans are flagged more often than Democrats for sharing misinformation on X’s Community Notes" by Renault et al.
    scores      = pl.read_parquet("data/2026-02-03-scored_notes.parquet")     
    scores      = enrich_with_intercepts_and_factors(scores)
    topics      = pl.read_parquet("data/from-soham-notes_full.parquet")
    logger.info("Data loaded successfully")
        
    # Enrollment data not in users file; find the date each user made their first action
    first_note_written      = notes     .group_by("noteAuthorParticipantId").agg(createdAtMillis=pl.col("createdAtMillis").min())
    first_note_rated        = ratings   .group_by("raterParticipantId")     .agg(createdAtMillis=pl.col("createdAtMillis").min())
    first_note_requested    = requests  .group_by("requesterParticipantId") .agg(createdAtMillis=pl.col("createdAtMillis").min())
    first_action = (
        pl.concat([
            first_note_written  .rename({"noteAuthorParticipantId": "participantId"}),
            first_note_rated    .rename({"raterParticipantId":      "participantId"}),
            first_note_requested.rename({"requesterParticipantId":  "participantId"}),
        ])
        .group_by("participantId")
        .agg(participantFirstActionMillis=pl.col("createdAtMillis").min())
    )
    logger.info(f"First action calculated for {len(first_action):,} users")

    # Join anything outside to top 5 topics into "other"
    top_5_topics = ["sports", "diaries_&_daily_life", "business_&_entrepreneurs", "science_&_technology", "news_&_social_concern"]
    topics = topics.with_columns(condensed_topic=pl.when(pl.col("topic").is_in(top_5_topics)).then(pl.col("topic")).otherwise(pl.lit("other")))
    topics = topics.select("noteId", "topic", "condensed_topic")
    logger.info("Topics processed and enriched with condensed topic")

    # Calculate whether a note ever achieved CRH status
    note_ever_crh = (
        statuses
        .with_columns(status_time=pl.col("status_time").str.to_datetime("%Y-%m-%dT%H:%M:%S%.f%z"))
        .filter(pl.col("status_time") <= pl.datetime(2026, 2, 3, time_zone="UTC"))
        .with_columns(Crh = pl.col("status") == "CURRENTLY_RATED_HELPFUL")
        .group_by("note_id")
        .agg(noteEverCrh = pl.col("Crh").any())
    )
    post_ever_crh = (
        notes
        .select("noteId", "tweetId")
        .join(note_ever_crh, left_on="noteId", right_on="note_id", how="left", validate="1:1")
        .group_by("tweetId")
        .agg(postEverCrh=pl.col("noteEverCrh").fill_null(False).any())
    )
    logger.info(f"Calculated CRH statuses for {len(note_ever_crh):,} notes")
    logger.info(f"Calculated CRH statuses for {len(post_ever_crh):,} posts")

    # Enrich with scores and factors
    notes = notes.join(scores, left_on="noteId", right_on="noteId", how="left", coalesce=True, validate="1:1")
    ratings = ratings.join(scores, left_on="noteId", right_on="noteId", how="left", coalesce=True, validate="m:1")
    logger.info("Enriched notes and ratings with scores and factors")

    # Enrich with CRH statuses
    notes    = notes   .join(note_ever_crh, left_on="noteId", right_on="note_id", how="left", coalesce=True, validate="1:1")
    ratings  = ratings .join(note_ever_crh, left_on="noteId", right_on="note_id", how="left", coalesce=True, validate="m:1")
    requests = requests.join(post_ever_crh, on="tweetId", how="left", validate="m:1")
    logger.info("Enriched notes and ratings with CRH statuses")

    # Enrich with topics
    notes    = notes   .join(topics, on="noteId", how="left", validate="1:1") # TODO: Get more recent data from soham
    ratings  = ratings .join(topics, on="noteId", how="left", validate="m:1")
    logger.info("Enriched notes and ratings with topics")
    # TODO: Topics for note requests? 

    # Enrich with user join dates
    notes    = notes   .join(first_action.rename({"participantId": "noteAuthorParticipantId"}),on="noteAuthorParticipantId",   how="left", validate="m:1")
    ratings  = ratings .join(first_action.rename({"participantId": "raterParticipantId"}),     on="raterParticipantId",        how="left", validate="m:1")
    requests = requests.join(first_action.rename({"participantId": "requesterParticipantId"}), on="requesterParticipantId",    how="left", validate="m:1")
    logger.info("Enriched notes and ratings with user join dates")

    notes    = _enrich_with_user_and_calendar_month(notes)
    ratings  = _enrich_with_user_and_calendar_month(ratings)
    requests = _enrich_with_user_and_calendar_month(requests)

    ratings  = ratings   .with_columns(ratingDate= pl.from_epoch(pl.col("createdAtMillis"), time_unit="ms").dt.date())
    requests = requests  .with_columns(requestDate=pl.from_epoch(pl.col("createdAtMillis"), time_unit="ms").dt.date())
    logger.info("Calculated user months and calendar months")

    # Enrich with Renault partisanship labels
    notes    = notes     .join(partisanship.select("note_id", "party").rename({"party":"postAuthorParty"}), left_on="noteId", right_on="note_id", coalesce=True, how="left", validate="1:1")
    ratings  = ratings   .join(partisanship.select("note_id", "party").rename({"party":"postAuthorParty"}), left_on="noteId", right_on="note_id", coalesce=True, how="left", validate="m:1")
    # TODO: Partisanship for note requests? 
    logger.info("Enriched notes and ratings with partisanship labels")

    # Enrich ratings with note-level data (e.g. final factor, final intercept, num ratings, CRH status, topic, etc.)
    ratings = ratings.join(
        notes.select("noteId", "noteEverCrh", "noteFinalFactor", "noteFinalIntercept", "topic", "postAuthorParty", "classification"),
        on="noteId",
        how="left",
        validate="m:1"
    )
    logger.info("Enriched ratings with note-level data")

    # Calculate whether request resulted in note, and if so, whether that note achieved CRH status
    request_outcomes = (
        notes
        .select("tweetId", "noteEverCrh")
        .group_by("tweetId")
        .agg(requestResultedInNote=pl.lit(True), requestResultedInCrh=pl.col("noteEverCrh").any())
    )
    requests = requests.join(request_outcomes, left_on="tweetId", right_on="tweetId", how="left", coalesce=True, validate="m:1")
    requests = requests.with_columns(
        requestResultedInNote=pl.col("requestResultedInNote").fill_null(False),
        requestResultedInCrh =pl.col("requestResultedInCrh") .fill_null(False)
    )
    logger.info("Enriched requests with outcomes")

    # Aggregate all users' notes per month
    user_notes = notes.group_by(["noteAuthorParticipantId", "userMonth"]).agg(
        calendarMonth=pl.col("calendarMonth").first(),
        notesCreated=pl.len(),
        hitRate=pl.col("noteEverCrh").mean(),
        hits=pl.col("noteEverCrh").sum(),
        avgNoteFactor=pl.col("noteFinalFactor").mean(),
        avgNoteIntercept=pl.col("noteFinalIntercept").mean(),
        topicsTargeted=pl.col("topic").filter(pl.col("topic").is_not_null()).n_unique(),
        avgRatingsEarned=pl.col("numRatings").mean(),
        *[
            pl.col("condensed_topic")
            .filter(pl.col("condensed_topic") == topic)
            .count()
            .alias(f"{topic}Count")
            for topic in top_5_topics + ["other"]
        ]
    ).sort("noteAuthorParticipantId", "userMonth")
    logger.info(f"Aggregated user notes: {len(user_notes):,} rows")

    # Aggregate all users' ratings per month
    user_ratings = ratings.group_by(["raterParticipantId", "userMonth"]).agg(
        calendarMonth=pl.col("calendarMonth").first(),
        notesRated=pl.len(),
        avgHelpfulFactor=pl.col("noteFinalFactor").filter(pl.col("helpfulnessLevel") == "HELPFUL").mean(),
        avgNotHelpfulFactor=pl.col("noteFinalFactor").filter(pl.col("helpfulnessLevel") != "HELPFUL").mean(),
        avgHelpfulIntercept=pl.col("noteFinalIntercept").filter(pl.col("helpfulnessLevel") == "HELPFUL").mean(),
        avgNotHelpfulIntercept=pl.col("noteFinalIntercept").filter(pl.col("helpfulnessLevel") != "HELPFUL").mean(),
        correctHelpfuls=pl.col("noteEverCrh").filter(pl.col("helpfulnessLevel") == "HELPFUL").sum(),
        correctNotHelpfuls=(~pl.col("noteEverCrh")).filter(pl.col("helpfulnessLevel") != "HELPFUL").sum(),

        # Counts by factor sign x helpfulness
        posFactorRatedHelpful=((pl.col("noteFinalFactor") > 0) & (pl.col("helpfulnessLevel") == "HELPFUL")).sum(),
        posFactorRatedNotHelpful=((pl.col("noteFinalFactor") > 0) & (pl.col("helpfulnessLevel") != "HELPFUL")).sum(),
        negFactorRatedHelpful=((pl.col("noteFinalFactor") < 0) & (pl.col("helpfulnessLevel") == "HELPFUL")).sum(),
        negFactorRatedNotHelpful=((pl.col("noteFinalFactor") < 0) & (pl.col("helpfulnessLevel") != "HELPFUL")).sum(),

        # % correct among +/- factor notes rated helpful/not helpful
        pctCorrectPosFactorHelpful=pl.col("noteEverCrh").filter((pl.col("noteFinalFactor") > 0) & (pl.col("helpfulnessLevel") == "HELPFUL")).mean(),
        pctCorrectPosFactorNotHelpful=(~pl.col("noteEverCrh")).filter((pl.col("noteFinalFactor") > 0) & (pl.col("helpfulnessLevel") != "HELPFUL")).mean(),
        pctCorrectNegFactorHelpful=pl.col("noteEverCrh").filter((pl.col("noteFinalFactor") < 0) & (pl.col("helpfulnessLevel") == "HELPFUL")).mean(),
        pctCorrectNegFactorNotHelpful=(~pl.col("noteEverCrh")).filter((pl.col("noteFinalFactor") < 0) & (pl.col("helpfulnessLevel") != "HELPFUL")).mean(),

        # % correct among helpful/not-helpful ratings overall
        pctHelpfulRatingsCorrect=pl.col("noteEverCrh").filter(pl.col("helpfulnessLevel") == "HELPFUL").mean(),
        pctNotHelpfulRatingsCorrect=(~pl.col("noteEverCrh")).filter(pl.col("helpfulnessLevel") != "HELPFUL").mean(),

        uniqueDaysRated=pl.col("ratingDate").n_unique(),
        avgPostsRatedPerDay=pl.len() / pl.col("ratingDate").n_unique(),
        uniqueTopicsRated=pl.col("topic").filter(pl.col("topic").is_not_null()).n_unique(),

        # Classifications from "Hyperactive Minority Alter the Stability of Community Notes" by Nudo et al.
        antiDemNNRatings    =((pl.col("postAuthorParty") == "democrat")   & (pl.col("classification") == "MISINFORMED_OR_POTENTIALLY_MISLEADING") & (pl.col("helpfulnessLevel") == "HELPFUL")).sum(),
        proDemNNRatings     =((pl.col("postAuthorParty") == "democrat")   & (pl.col("classification") == "MISINFORMED_OR_POTENTIALLY_MISLEADING") & (pl.col("helpfulnessLevel") != "HELPFUL")).sum(),
        proDemNNNRatings    =((pl.col("postAuthorParty") == "democrat")   & (pl.col("classification") == "NOT_MISLEADING")                        & (pl.col("helpfulnessLevel") == "HELPFUL")).sum(),
        antiDemNNNRatings   =((pl.col("postAuthorParty") == "democrat")   & (pl.col("classification") == "NOT_MISLEADING")                        & (pl.col("helpfulnessLevel") != "HELPFUL")).sum(),
        antiRepNNRatings    =((pl.col("postAuthorParty") == "republican") & (pl.col("classification") == "MISINFORMED_OR_POTENTIALLY_MISLEADING") & (pl.col("helpfulnessLevel") == "HELPFUL")).sum(),
        proRepNNRatings     =((pl.col("postAuthorParty") == "republican") & (pl.col("classification") == "MISINFORMED_OR_POTENTIALLY_MISLEADING") & (pl.col("helpfulnessLevel") != "HELPFUL")).sum(),
        proRepNNNRatings    =((pl.col("postAuthorParty") == "republican") & (pl.col("classification") == "NOT_MISLEADING")                        & (pl.col("helpfulnessLevel") == "HELPFUL")).sum(),
        antiRepNNNRatings   =((pl.col("postAuthorParty") == "republican") & (pl.col("classification") == "NOT_MISLEADING")                        & (pl.col("helpfulnessLevel") != "HELPFUL")).sum(),
        *[
            pl.col("condensed_topic")
            .filter(pl.col("condensed_topic") == topic)
            .count()
            .alias(f"{topic}RatedCount")
            for topic in top_5_topics + ["other"]
        ],
    ).with_columns(
        overallAccuracy=(pl.col("correctHelpfuls") + pl.col("correctNotHelpfuls")) / pl.col("notesRated"),
        helpfulNotHelpfulFactorDiff=pl.col("avgHelpfulFactor") - pl.col("avgNotHelpfulFactor"),
        helpfulNotHelpfulInterceptDiff=pl.col("avgHelpfulIntercept") - pl.col("avgNotHelpfulIntercept"),
        proDemRatings=pl.col("proDemNNRatings") + pl.col("proDemNNNRatings"),
        antiDemRatings=pl.col("antiDemNNRatings") + pl.col("antiDemNNNRatings"),
        proRepRatings=pl.col("proRepNNRatings") + pl.col("proRepNNNRatings"),
        antiRepRatings=pl.col("antiRepNNRatings") + pl.col("antiRepNNNRatings"),
    ).sort("raterParticipantId", "userMonth")
    logger.info(f"Aggregated user ratings: {len(user_ratings):,} rows")

    user_requests = requests.group_by(["requesterParticipantId", "userMonth"]).agg(
        calendarMonth=pl.col("calendarMonth").first(),
        requestsMade=pl.len(),
        numRequestsResultingInCrh   = pl.col("requestResultedInCrh") .sum(),
        numRequestsResultingInNote  = pl.col("requestResultedInNote").sum(),
        pctRequestResultedInNote    = pl.col("requestResultedInNote").mean(),
        pctRequestResultedInCrh     = pl.col("requestResultedInCrh") .mean(),
    ).sort("requesterParticipantId", "userMonth")
    logger.info(f"Aggregated user requests: {len(user_requests):,} rows")

    # TODO: Number of ratings sessions + Average number of posts rated per session

    # Write
    user_notes.write_parquet("data/user_note_traj.parquet")
    user_ratings.write_parquet("data/user_rating_traj.parquet")
    user_requests.write_parquet("data/user_request_traj.parquet")
    logger.info("Wrote full trajectory files")

    # Sample 20,000 users
    all_user_ids = first_action.select("participantId").unique().sort("participantId")
    hash = md5("".join(all_user_ids["participantId"]).encode("utf-8")).hexdigest()
    logger.info(f"Hash of all user ids: {hash}") # For reproducibility checks
    sampled_user_ids = all_user_ids.sample(20_000, seed=465309)
    sampled_user_notes = user_notes.join(sampled_user_ids, left_on="noteAuthorParticipantId", right_on="participantId", how="inner")
    sampled_user_notes.write_parquet("data/sample_user_note_traj.parquet")
    sampled_user_ratings = user_ratings.join(sampled_user_ids, left_on="raterParticipantId", right_on="participantId", how="inner")
    sampled_user_ratings.write_parquet("data/sample_user_rating_traj.parquet")
    sampled_user_requests = user_requests.join(sampled_user_ids, left_on="requesterParticipantId", right_on="participantId", how="inner")
    sampled_user_requests.write_parquet("data/sample_user_request_traj.parquet")
    logger.info(
        "Wrote sampled trajectory files. Sampled 20_000 users. "
        f"{len(sampled_user_notes):,} user-months with notes from {len(sampled_user_notes['noteAuthorParticipantId'].unique()):,} unique note authors, and "
        f"{len(sampled_user_ratings):,} user-months with ratings from {len(sampled_user_ratings['raterParticipantId'].unique()):,} unique raters."
        f"{len(sampled_user_requests):,} user-months with requests from {len(sampled_user_requests['requesterParticipantId'].unique()):,} unique requesters."
    )
