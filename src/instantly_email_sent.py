# app/instantly_email_sent.py

from airtable_service import PodcastService
from datetime import datetime

def update_airtable_when_email_sent(data):
    # Extract necessary data from webhook payload
    campaign_id = data.get('campaign_id')
    airID = data.get('airID')

    if not campaign_id or not airID:
        print("campaign_id or airID not provided in webhook data")
        return

    # Initialize Airtable service
    airtable_service = PodcastService()

    CAMPAIGN_MANAGER_TABLE_NAME = 'Campaign Manager'  # Adjust the name if needed
    CAMPAIGNS_TABLE_NAME = 'Campaigns'

    # Step 1: Search for the campaign record 
    formula = "{Instantly} = '{}'".format(campaign_id)
    search_results = airtable_service.search_records(CAMPAIGNS_TABLE_NAME, formula, view=None)
    

    if not search_results:
        print("No matching campaign found.")
        # Depending on your workflow, you might want to stop here or proceed
        # return

    # Step 2: Get the campaign manager record using 'airID'
    campaign_manager_record = airtable_service.get_record(CAMPAIGN_MANAGER_TABLE_NAME, airID)

    if not campaign_manager_record:
        print(f"No record found with id {airID}")
        return

    fields = campaign_manager_record.get('fields', {})
    correspondence = fields.get('Correspondence', '')
    outreach_date = fields.get('Outreach Date')

    # Extract additional data from webhook payload
    timestamp = data.get('timestamp', datetime.now().isoformat())
    event_type = data.get('event_type', '')
    personalization = data.get('personalization', '')

    # Append new correspondence entry
    new_entry = f"\n\nDate: {timestamp}\nEvent Type: {event_type}\nMessage: {personalization}"
    updated_correspondence = (correspondence or '') + new_entry

    # Prepare fields to update
    today_date = datetime.now().strftime('%Y-%m-%d')
    fields_to_update = {
        'Correspondence': updated_correspondence,
        'Outreach Date': today_date
    }

    # If 'Outreach Date' does not exist, set 'Status' to 'Outreached'
    if not outreach_date:
        fields_to_update['Status'] = 'Outreached'

    # Update the record in Airtable   update_record(self, table_name, record_id, fields)
    updated_record = airtable_service.update_record(CAMPAIGN_MANAGER_TABLE_NAME, airID, fields_to_update)

    if updated_record:
        print(f"Record {airID} updated successfully.")
    else:
        print(f"Failed to update record {airID}")