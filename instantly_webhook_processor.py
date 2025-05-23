# instantly_webhook_processor.py

import json
from datetime import datetime

# Assuming instantly_leads_db.py is in the same directory or accessible in PYTHONPATH
# If it's in 'src' and this script is also in 'src', use: from .instantly_leads_db import ...
# If this script is in root and instantly_leads_db.py is in src, adjust path or ensure src is in PYTHONPATH.
# For simplicity, if both are in the same dir or root:
from instantly_leads_db import add_instantly_lead_record, update_instantly_lead_record, get_instantly_lead_by_id

# Note: In a production environment, this script would typically be part of a web application (e.g., using Flask or FastAPI)
# that exposes an HTTP endpoint for Instantly.ai to send webhook POST requests to.

def process_instantly_webhook(webhook_data: dict):
    """Processes incoming webhook data from Instantly.ai.

    Args:
        webhook_data (dict): The JSON payload received from Instantly.ai.
    """
    print(f"\nReceived webhook data: {json.dumps(webhook_data, indent=2)}")

    event_type = webhook_data.get("event") # Assuming Instantly provides an 'event' type field

    if not event_type:
        print("Error: Webhook data does not contain an 'event' type field.")
        return False

    print(f"Processing event type: {event_type}")

    # --- Event: Lead Created (Hypothetical) ---
    # Assuming a 'lead.created' event where 'data' contains the full lead object
    if event_type == "lead.created":
        lead_data = webhook_data.get("data")
        if not lead_data or not isinstance(lead_data, dict) or not lead_data.get("id"):
            print("Error: 'lead.created' event data is missing, not a dictionary, or missing lead id.")
            return False
        
        print(f"Attempting to add new lead: {lead_data.get('id')} - {lead_data.get('email')}")
        # Before adding, ensure it's not a duplicate if webhooks can sometimes send create for existing
        existing_lead = get_instantly_lead_by_id(lead_data.get('id'))
        if existing_lead:
            print(f"Lead {lead_data.get('id')} already exists in backup. Attempting update instead.")
            # You might want to adapt the lead_data to fit what update_instantly_lead_record expects
            # For simplicity, let's assume the full lead_data can be used to derive update_data
            # This would require mapping all fields from lead_data to their snake_case db column names.
            # For now, just calling update with a limited set of fields or the whole thing if mapping is identical.
            # Example: a real update payload might only contain changed fields.
            # We are passing the full lead_data, update function should handle it.
            # Create a dictionary of fields that are not None to update
            update_payload = {k: v for k, v in lead_data.items() if v is not None and k != 'id'}
             # Map API field names to DB column names for the update_data dictionary
            db_update_payload = map_api_to_db_fields(lead_data) # You'd need to implement this mapping
            if db_update_payload.get('lead_id'): # remove lead_id as it's used in where clause
                del db_update_payload['lead_id']
            
            if update_instantly_lead_record(lead_data.get('id'), db_update_payload):
                 print(f"Successfully updated existing lead {lead_data.get('id')} from 'lead.created' event.")
                 return True
            else:
                 print(f"Failed to update existing lead {lead_data.get('id')} from 'lead.created' event.")
                 return False
        else:
            if add_instantly_lead_record(lead_data):
                print(f"Successfully added new lead {lead_data.get('id')} to backup.")
                return True
            else:
                print(f"Failed to add new lead {lead_data.get('id')} to backup.")
                return False

    # --- Event: Lead Updated (Hypothetical) ---
    # Assuming a 'lead.updated' event where 'data' contains the lead_id and 'changes' (a dict of updated fields)
    # Or, it might send the full updated lead object.
    # Let's assume it sends the full updated lead object for simplicity here.
    elif event_type == "lead.updated":
        lead_data = webhook_data.get("data") # Assuming full lead object is sent
        if not lead_data or not isinstance(lead_data, dict) or not lead_data.get("id"):
            print("Error: 'lead.updated' event data is missing, not a dict, or missing lead id.")
            return False

        lead_id = lead_data.get("id")
        print(f"Attempting to update lead: {lead_id}")

        # The lead_data IS the data from Instantly, so it has API field names.
        # We need to map these to our database's snake_case column names.
        update_payload_for_db = map_api_to_db_fields(lead_data)
        
        # Remove lead_id from the update payload as it's used in the WHERE clause of the update function
        # and shouldn't be in the SET part.
        if 'lead_id' in update_payload_for_db:
            del update_payload_for_db['lead_id'] 

        if not update_payload_for_db: # if mapping results in empty dict (e.g. only id was present)
            print(f"No fields to update for lead {lead_id} after mapping.")
            return True # Or False, depending on desired behavior for no-op updates

        if update_instantly_lead_record(lead_id, update_payload_for_db):
            print(f"Successfully updated lead {lead_id} in backup.")
            return True
        else:
            # If update failed, it could be because the lead doesn't exist yet in backup.
            # This might happen if 'lead.updated' event arrives before a batch backup has run.
            print(f"Failed to update lead {lead_id}. Attempting to add as new lead.")
            if add_instantly_lead_record(lead_data): # add_instantly_lead_record expects API field names
                print(f"Successfully added lead {lead_id} from 'lead.updated' event as it was missing.")
                return True
            else:
                print(f"Failed to add lead {lead_id} from 'lead.updated' event after update failed.")
                return False
    
    # --- Event: Lead Status Changed (More specific update example) ---
    elif event_type == "lead.status.changed": # Hypothetical specific event
        event_payload = webhook_data.get("data")
        if not event_payload or not isinstance(event_payload, dict):
            print("Error: 'lead.status.changed' event data is missing or not a dictionary.")
            return False
        
        lead_id = event_payload.get("lead_id") # Assuming event payload has lead_id directly
        new_status = event_payload.get("new_status") # And the new status

        if not lead_id or new_status is None: # new_status could be 0 which is valid
            print("Error: Missing lead_id or new_status in 'lead.status.changed' event data.")
            return False

        print(f"Attempting to update status for lead: {lead_id} to {new_status}")
        update_fields = {
            "lead_status": new_status,
            "timestamp_updated": datetime.utcnow().isoformat() + "Z" # Update timestamp_updated
        }
        if update_instantly_lead_record(lead_id, update_fields):
            print(f"Successfully updated status for lead {lead_id}.")
            return True
        else:
            print(f"Failed to update status for lead {lead_id}.")
            # Consider if you need to add if not found, like in the generic 'lead.updated'
            return False

    else:
        print(f"Unrecognized or unhandled event type: {event_type}")
        return False
    
    return False # Default case if no event was successfully processed

def map_api_to_db_fields(lead_api_data: dict) -> dict:
    """Maps keys from Instantly API lead data to database column names (snake_case).
       Also handles JSONB fields by ensuring they are passed as dicts.
    """
    if not lead_api_data:
        return {}
    
    # This mapping should match the one in add_instantly_lead_record's insert_data
    db_data = {
        "lead_id": lead_api_data.get("id"),
        "timestamp_created": lead_api_data.get("timestamp_created"),
        "timestamp_updated": lead_api_data.get("timestamp_updated"),
        "organization_id": lead_api_data.get("organization"),
        "lead_status": lead_api_data.get("status"),
        "email_open_count": lead_api_data.get("email_open_count"),
        "email_reply_count": lead_api_data.get("email_reply_count"),
        "email_click_count": lead_api_data.get("email_click_count"),
        "company_domain": lead_api_data.get("company_domain"),
        "status_summary": lead_api_data.get("status_summary"), 
        "campaign_id": lead_api_data.get("campaign"),
        "email": lead_api_data.get("email"),
        "personalization": lead_api_data.get("personalization"),
        "website": lead_api_data.get("website"),
        "last_name": lead_api_data.get("last_name"),
        "first_name": lead_api_data.get("first_name"),
        "company_name": lead_api_data.get("company_name"),
        "phone": lead_api_data.get("phone"),
        "payload": lead_api_data.get("payload"), 
        "status_summary_subseq": lead_api_data.get("status_summary_subseq"),
        "last_step_from": lead_api_data.get("last_step_from"),
        "last_step_id": lead_api_data.get("last_step_id"),
        "last_step_timestamp_executed": lead_api_data.get("last_step_timestamp_executed"),
        "email_opened_step": lead_api_data.get("email_opened_step"),
        "email_opened_variant": lead_api_data.get("email_opened_variant"),
        "email_replied_step": lead_api_data.get("email_replied_step"),
        "email_replied_variant": lead_api_data.get("email_replied_variant"),
        "email_clicked_step": lead_api_data.get("email_clicked_step"),
        "email_clicked_variant": lead_api_data.get("email_clicked_variant"),
        "lt_interest_status": lead_api_data.get("lt_interest_status"),
        "subsequence_id": lead_api_data.get("subsequence_id"),
        "verification_status": lead_api_data.get("verification_status"),
        "pl_value_lead": lead_api_data.get("pl_value_lead"),
        "timestamp_added_subsequence": lead_api_data.get("timestamp_added_subsequence"),
        "timestamp_last_contact": lead_api_data.get("timestamp_last_contact"),
        "timestamp_last_open": lead_api_data.get("timestamp_last_open"),
        "timestamp_last_reply": lead_api_data.get("timestamp_last_reply"),
        "timestamp_last_interest_change": lead_api_data.get("timestamp_last_interest_change"),
        "timestamp_last_click": lead_api_data.get("timestamp_last_click"),
        "enrichment_status": lead_api_data.get("enrichment_status"),
        "list_id": lead_api_data.get("list_id"),
        "last_contacted_from": lead_api_data.get("last_contacted_from"),
        "uploaded_by_user": lead_api_data.get("uploaded_by_user"),
        "upload_method": lead_api_data.get("upload_method"),
        "assigned_to": lead_api_data.get("assigned_to"),
        "is_website_visitor": lead_api_data.get("is_website_visitor"),
        "timestamp_last_touch": lead_api_data.get("timestamp_last_touch"),
        "esp_code": lead_api_data.get("esp_code")
    }
    # Return a new dict with None values removed, as update_instantly_lead_record expects only fields to change.
    # However, for a full object mapping, None is fine if the DB column allows NULL.
    # For an update, we only want to send fields that are present in the API data.
    return {k: v for k, v in db_data.items() if v is not None}


if __name__ == "__main__":
    print("Instantly Webhook Processor Script (Test Mode)")
    print("---------------------------------------------")

    # Example 1: Hypothetical 'lead.created' event
    sample_new_lead_event = {
        "event": "lead.created",
        "data": { # This structure should match the lead object from Instantly API
            "id": "a1b2c3d4-feed-face-cafe-0123456789ab", # New UUID
            "timestamp_created": datetime.utcnow().isoformat() + "Z",
            "timestamp_updated": datetime.utcnow().isoformat() + "Z",
            "organization": "org_uuid_example",
            "status": 1, # Active
            "email": "new.lead@example.com",
            "first_name": "New",
            "last_name": "Lead",
            "company_name": "Fresh Prospects Inc.",
            "campaign": "campaign_uuid_123",
            "email_open_count": 0,
            "email_reply_count": 0,
            "email_click_count": 0,
            "payload": {"custom_field_1": "webhook_test_value"}
            # ... other relevant lead fields
        }
    }
    print("\n--- Testing 'lead.created' event ---")
    process_instantly_webhook(sample_new_lead_event)

    # Example 2: Hypothetical 'lead.updated' event (sending full lead object)
    # Ensure this lead_id exists in your backup if you want the update part to succeed first.
    # Otherwise, it will try to add it as new.
    sample_updated_lead_full_event = {
        "event": "lead.updated",
        "data": {
            "id": "a1b2c3d4-feed-face-cafe-0123456789ab", # Same ID as above for testing update/add logic
            "timestamp_updated": datetime.utcnow().isoformat() + "Z",
            "status": 2, # Paused
            "email_open_count": 5,
            "first_name": "NewName", # Changed field
            # other fields might be present, even if unchanged
            "email": "new.lead@example.com", 
            "last_name": "Lead",
            "company_name": "Fresh Prospects Inc.",
            "campaign": "campaign_uuid_123",
            "payload": {"custom_field_1": "webhook_updated_value", "new_info": "added by update"}
        }
    }
    print("\n--- Testing 'lead.updated' event (with full lead data) ---")
    process_instantly_webhook(sample_updated_lead_full_event)

    # Example 3: Hypothetical 'lead.status.changed' event (more specific)
    sample_status_change_event = {
        "event": "lead.status.changed",
        "data": {
            "lead_id": "a1b2c3d4-feed-face-cafe-0123456789ab", # Same ID
            "new_status": -1, # Bounced
            "timestamp_event": datetime.utcnow().isoformat() + "Z"
        }
    }
    print("\n--- Testing 'lead.status.changed' event ---")
    process_instantly_webhook(sample_status_change_event)

    print("\nWebhook processing tests complete.")
    print("Note: Actual webhook payloads from Instantly.ai may vary. Adapt parsing as needed.") 