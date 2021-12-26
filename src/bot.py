import os
import time
import json
from dotenv import load_dotenv
from huggingface_hub.inference_api import InferenceApi
import tweepy
from preprocessing import preprocess_tweet


def getStatusId(url: str) -> int:
    return int(url.split("/")[-1])


def getUsername(url: str) -> str:
    return url.split("/")[3]


load_dotenv()
RUN_EVERY_NSECS = int(os.getenv("RUN_EVERY_NSECS"))
# twitter api
TWITTER_API_KEY = os.getenv("TWITTER_API_KEY")
TWITTER_API_SECRET = os.getenv("TWITTER_API_SECRET")
TWITTER_ACCESS_TOKEN = os.getenv("TWITTER_ACCESS_TOKEN")
TWITTER_ACCESS_SECRET = os.getenv("TWITTER_ACCESS_SECRET")
# hugging face api
HF_API_TOKEN = os.getenv("HF_TOKEN")
# REPO_ID = "bhadresh-savani/distilbert-base-uncased-emotion"
# REPO_ID = "j-hartmann/emotion-english-distilroberta-base"
REPO_ID = "finiteautomata/bertweet-base-emotion-analysis"


def main():
    # Authenticate access
    auth = tweepy.OAuthHandler(TWITTER_API_KEY, TWITTER_API_SECRET)
    auth.set_access_token(TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_SECRET)
    # Create API handler
    api = tweepy.API(auth)

    inference = InferenceApi(repo_id=REPO_ID, token=HF_API_TOKEN)

    cutoff_time = round((time.time() - RUN_EVERY_NSECS) * 1000)
    duplicate_tweets = set()
    print(f"INFO: LAMBDA STARTED: cutoff_time={cutoff_time}")
    msgs = api.get_direct_messages(count=50)
    retweets = []

    for msg in msgs:
        if int(msg.created_timestamp) < cutoff_time:
            break
        print(f"INFO: PROCESING DM: msg={msg._json}")
        if msg.type == "message_create":
            url_list = msg.message_create["message_data"]["entities"]["urls"]
            if len(url_list):
                tweet_url = url_list[0]["expanded_url"]
                status_id = getStatusId(tweet_url)

                if status_id in duplicate_tweets:
                    print("INFO: DUPLICATE: Tweet already seen")
                    continue
                duplicate_tweets.add(status_id)

                tweet = api.get_status(
                    status_id, include_my_retweet=True, tweet_mode="extended"
                )
                # submitter = api.get_user(user_id=msg.message_create["sender_id"])

                text = preprocess_tweet(tweet.full_text, lang="en")
                prediction = inference(inputs=text)[0]
                print(f"INFO: HF INFERENCE: prediction={prediction}")
                prediction[0]["label"] = "neutral"
                prediction_text = "\n".join(
                    [f"{d['label']}: {round(100*d['score'], 2)}%" for d in prediction]
                )

                rt_text = "\n".join(
                    [
                        "Prediction",
                        prediction_text,
                        # f"\nSubmission by @{submitter.screen_name}",
                    ]
                )
                try:
                    status = api.update_status(rt_text, attachment_url=tweet_url)
                    print(f"SUCCESS: RETWEETED: id={status.id}")
                    retweets.append(status.id)
                except Exception as err:
                    # 403 response when already retweeted with same content
                    # given same inference model, resulting in same predictions for same text
                    # avoids retweeting already retweeted tweets whose DM << cutoff_time
                    print(f"ERROR: {err}")
                break
            else:
                print(f"INFO: Message does not have tweet")
    return retweets


def lambda_handler(event, context):
    retweets = main()
    return {"statusCode": 200, "body": json.dumps(retweets)}


if __name__ == "__main__":
    lambda_handler(None, None)
