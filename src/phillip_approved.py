from airtable_service import PodcastService
import pandas as pd


CSV_FILE_PATH = "C:/Users/ebube/Downloads/Phillip Swan - Potential Podcasts.csv"
PODCAST_NAME = "Podcast"
airtable_service = PodcastService()
df = pd.read_csv(CSV_FILE_PATH)

for podcast in df[PODCAST_NAME]:
    try:
        formula = f"AND({PODCAST_NAME} = '{podcast}')"
        
        cm_record = airtable_service.search_records("Campaign Manager", formula=formula, view="Phillip Podcasts")
        if cm_record:
            print(f"Campaign Manager record found for {podcast}")
            cm_record_id = cm_record[0]["id"]
            cm_record_fields = cm_record[0]["fields"]
            airtable_service.update_record("Campaign Manager", cm_record_id, {"Approved": True})
            print(f"Campaign Manager record updated for {podcast}")
        else:
            print(f"Campaign Manager record not found for {podcast}")
    except Exception as e:
        print(f"Error: {e}")




