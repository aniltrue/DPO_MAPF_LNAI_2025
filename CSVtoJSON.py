from nltk import word_tokenize
import numpy as np
import pandas as pd
import json
import textblob

df_real = pd.read_excel("NLP Data Updated.xlsx")
df_nego_data = pd.read_csv("NegoData.csv")

output = {}
session = {}
removed_counter = 0
round_counter = 0
total_removed = 0

for i, row in df_nego_data.iterrows():
    if row["Round"] == 0 and not row["IsUser"]:
        session = {"SessionID": row["SessionID"], "AgentName": row["AgentName"], "History": []}
        removed_counter = 0
        round_counter = 0

    if len(session["History"]) > row["Round"]:
        continue

    is_removed = False

    row_round = df_nego_data.loc[
        (df_nego_data["SessionID"] == row["SessionID"]) & (df_nego_data["Round"] == row["Round"])]
    row_user = row_round.loc[(row_round["IsUser"] == 1)].iloc[0]
    row_agent = row_round.loc[(row_round["IsUser"] == 0)]

    if len(row_agent) < 1:
        row_agent = None
    else:
        row_agent = row_agent.iloc[0]

    # Find correct one
    user_text = " ".join(word_tokenize(row_user["NegotiationText"].lower().strip()))
    user_text = user_text.replace("ı", "i")

    row_real = df_real.loc[(df_real["Text"] == user_text)]

    if len(row_real) == 0:
        rows_real = df_real.loc[(df_real["Excited"] == row_user["Excited"])
                                & (df_real["Happy"] == row_user["Happy"]) & (df_real["Neutral"] == row_user["Neutral"])
                                & (df_real["Sad"] == row_user["Sad"]) & (df_real["Angry"] == row_user["Angry"])]

        if len(rows_real) == 0:
            removed_counter += 1
            round_counter += 1
            total_removed += 1

            is_removed = True

            row_real = {
                "Text": user_text,
                "Arg. Label": "Any",
                "Arg. Value": "0"
            }

        user_text = " ".join([textblob.Word(word).correct() for word in word_tokenize(user_text)]).lower().strip()

        row_real = df_real.loc[(df_real["Text"] == user_text)]

        if len(row_real) == 0:
            rows_real = rows_real.loc[(df_real["IssueAccommodation"] == row_user["IssueAccommodation"])
                                      & (df_real["IssueDestination"] == row_user["IssueDestination"])
                                      & (df_real["IssueSeason"] == row_user["IssueSeason"])
                                      & (df_real["IssueTransportation"] == row_user["IssueTransportation"])]

            if len(rows_real) == 0:
                removed_counter += 1
                round_counter += 1

                total_removed += 1

                is_removed = True

                row_real = {
                    "Text": user_text,
                    "Arg. Label": "Any",
                    "Arg. Value": "0"
                }
            else:
                best_score = 0.5
                best_match = None

                target_words = set(word_tokenize(user_text))

                for _, candidate in rows_real.iterrows():
                    candidate_words = word_tokenize(candidate["Text"].strip().lower())
                    candidate_words = set([textblob.Word(word).correct() for word in candidate_words])

                    jaccard_coef = len(target_words.intersection(candidate_words)) / len(
                        target_words.union(candidate_words))

                    if jaccard_coef > best_score:
                        best_score = jaccard_coef
                        best_match = candidate

                if best_match is None:
                    print("Could not be found:", user_text)

                    removed_counter += 1
                    round_counter += 1

                    total_removed += 1

                    row_real = {
                        "Text": user_text,
                        "Arg. Label": "Any",
                        "Arg. Value": "0"
                    }
                else:
                    row_real = best_match
        else:
            row_real = row_real.iloc[0]
    else:
        row_real = row_real.iloc[0]

    # print(user_text, row_real["Arg. Label"], row_real["Arg. Value"])

    # Generate Round
    nego_round = {
        "Round": row["Round"],
        "User": {
            "Text": " ".join(word_tokenize(row_real["Text"].strip().lower())),
            "BidContent": {
                "Accommodation": row_user["IssueAccommodation"],
                "Destination": row_user["IssueDestination"],
                "Season": row_user["IssueSeason"],
                "Transportation": row_user["IssueTransportation"],
            },
            "Utility": row_user["UserUtility"],
            "OppUtility": row_user["AgentUtility"],
            "Move": '' if str(row_user["Movement"]) == 'nan' else row_user["Movement"],
            "Emotion": {
                "Excited": row_user["Excited"],
                "Happy": row_user["Happy"],
                "Neutral": row_user["Neutral"],
                "Sad": row_user["Sad"],
                "Angry": row_user["Angry"]
            },
            "MajorEmotion": row_user["Emotion"],
            "Arguments": [
                {arg_label.replace("Acceptation", "Acceptance"): row_real["Arg. Value"].split(",")[j].replace(";", "-").replace("0", "")}
                for j, arg_label in enumerate(row_real["Arg. Label"].split(","))
            ],
            "MajorArgument": row_user["ArgumentType"],
            "IsAccept": bool(row_user["IsAccept"] > 0)
        },
        "Agent": {
            "Text": row_agent["NegotiationText"],
            "BidContent": {
                "Accommodation": row_agent["IssueAccommodation"],
                "Destination": row_agent["IssueDestination"],
                "Season": row_agent["IssueSeason"],
                "Transportation": row_agent["IssueTransportation"],
            },
            "Utility": row_agent["AgentUtility"],
            "OppUtility": row_agent["UserUtility"],
            "Move": '' if str(row_agent["Movement"]) == 'nan' else row_agent["Movement"],
            "MajorArgument": row_agent["ArgumentType"],
            "IsAccept": bool(row_agent["IsAccept"] > 0),
        } if row_agent is not None else None,
        "IsAgreement": bool(row_user["IsAccept"] + (row_agent["IsAccept"] if row_agent is not None else 0) > 0),
        "Minutes": row["Minutes"],
        "Seconds": row["Seconds"],
        "IsRemoved": is_removed
    }

    if row_agent is None:
        del nego_round["Agent"]

    session["History"].append(nego_round)

    round_counter += 1

    if row_user["IsAccept"]:
        session["Result"] = "Acceptance"
        session["WhoAccept"] = "User"
        session["UserFinalUtility"] = row_user["UserUtility"]
        session["AgentFinalUtility"] = row_user["AgentUtility"]
        session["Minutes"] = row["Minutes"]
        session["Seconds"] = row["Seconds"]

        if removed_counter == round_counter:
            print("Removed:", row["SessionID"])
        elif removed_counter > 0:
            print("Number of removed:", row["SessionID"], removed_counter, round_counter)

            output[row["SessionID"]] = session
        else:
            output[row["SessionID"]] = session

    elif row_agent is not None and row_agent["IsAccept"]:
        session["Result"] = "Acceptance"
        session["WhoAccept"] = "Agent"
        session["UserFinalUtility"] = row_agent["UserUtility"]
        session["AgentFinalUtility"] = row_agent["AgentUtility"]
        session["Minutes"] = row["Minutes"]
        session["Seconds"] = row["Seconds"]

        if removed_counter == round_counter:
            print("Removed:", row["SessionID"])
        elif removed_counter > 0:
            print("Number of removed:", row["SessionID"], removed_counter, round_counter)

            output[row["SessionID"]] = session
        else:
            output[row["SessionID"]] = session


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


with open("NegoNLPDataUpdated.json", "w", encoding='utf-8') as f:
    json.dump(output, f, cls=NpEncoder, indent=2, default=str)

print("Total Removed:", total_removed)
print("Total Found:", sum([len(session["History"]) for session in output.values()]))
print("In Real:", len(df_real))
print("In NegoData (Estimated):", len(df_nego_data) // 2)
