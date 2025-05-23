# Document Access Issues - Findings and Recommendations

## Issues Identified

1. **Invalid Document IDs**: Our testing revealed that the Google Docs document IDs extracted from the links in the Airtable records are no longer valid. When trying to access these documents, we receive `404 File not found` errors rather than `403 Permission denied` errors, indicating that the documents either don't exist or have been moved/deleted.

2. **Missing Parameter in API Call**: We identified and fixed a missing `data_type` parameter in the `transform_text_to_structured_data` function call within the `generate_structured_data` method.

## Test Results

- The service account successfully accesses the prompt documents (v1, v2, and keyword prompt).
- The client document IDs extracted from the Airtable record for Casey Cheshire return 404 errors.
- When using mock content as a workaround, the `generate_structured_data` function successfully generates bio and angles content.

## Recommendations

1. **Update Document Links in Airtable**: The document links stored in Airtable need to be updated with valid Google Docs links. The current links point to files that either don't exist or have been moved.

2. **Add Fallback Content**: Implement a fallback mechanism that uses example or placeholder content when document retrieval fails, similar to what we did in the test script.

3. **Improve Error Handling**: Enhance the error handling in the `get_content_from_docs` method to:
   - Detect 404 errors specifically
   - Log clear information about missing documents
   - Use a fallback approach when documents can't be accessed

4. **Document Verification Process**: Implement a process to periodically verify that document links in Airtable are still valid and accessible.

## Implementation Options

1. **Immediate Fix (Short-term)**:
   - Modify the `get_content_from_docs` method to handle document retrieval failures gracefully
   - Use mock/example content when documents can't be accessed
   - Log detailed information to help identify which records have invalid document links

2. **Sustainable Solution (Long-term)**:
   - Create a verification tool that checks all document links in Airtable
   - Update the Airtable records with valid links
   - Consider storing content snapshots in a database as a backup
   - Set up monitoring for document access issues

## Code Improvements Made

1. Fixed the missing `data_type` parameter in the `transform_text_to_structured_data` call:
```python
structured_data = await loop.run_in_executor(
    self.executor,
    lambda: self.openai_service.transform_text_to_structured_data(
        prompt="Parse out each of 3 bios and client angles, each angle has 3 parts: Topic, Outcome, Description", 
        raw_text=gemini_response,
        data_type="Structured",  # Added this missing parameter
        workflow="bio_and_angles"
    )
)
```

2. Created two diagnostic test scripts:
   - `test_google_docs_access.py` to test document access permissions
   - Updated `test_angles.py` to process records with mock content

## Next Steps

1. Implement the recommended fixes in the main `angles.py` file
2. Create a script to validate all document links in Airtable
3. Set up automated testing to catch similar issues before deployment 