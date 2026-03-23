# Deployment Guide for Human Experiment Portal

## 1. Hosting the Web App (GitHub Pages)
1. Commit all your latest changes in the `HumanExperimentConfig` folder to your GitHub repository.
2. Go to your repository settings on GitHub -> **Pages**.
3. Under **Build and deployment**, select **Deploy from a branch** and choose the `main` branch (or whichever branch your code is on) and the `/ (root)` folder.
4. Click **Save**. GitHub will automatically build and provide you with a URL (e.g., `https://yourusername.github.io/HumanExperimentConfig/`). This URL is your Portal.

## 2. Setting Up Google Apps Script (Backend)

The web app sends experiment data directly to a Google Sheet via a Google Apps Script Webhook.

1. Go to [Google Sheets](https://sheets.google.com/) and create a new blank spreadsheet.
2. In the menu, go to **Extensions > Apps Script**.
3. Replace the existing code with the following snippet:

```javascript
function doPost(e) {
  try {
    var ss = SpreadsheetApp.getActiveSpreadsheet();
    
    // Parse the incoming POST request
    var body = JSON.parse(e.postData.contents);
    var experimentId = body.experimentId;
    var timestamp = body.timestamp || new Date().toISOString();
    var data = body.data; // This is the runLog JSON string
    
    // Determine the target sheet based on experimentId
    var sheetName = "Unknown";
    if (experimentId === "DSB" || experimentId === "FIP" || experimentId === "TPB") {
      sheetName = experimentId;
    }
    
    var sheet = ss.getSheetByName(sheetName);
    
    // If the sheet doesn't exist, create it and add headers
    if (!sheet) {
      sheet = ss.insertSheet(sheetName);
      sheet.appendRow(["Timestamp", "Experiment ID", "Data (JSON)"]);
    }
    
    // Append the row with the data
    sheet.appendRow([timestamp, experimentId, data]);
    
    // Note: Due to 'no-cors' mode fetch, the browser will ignore the response, 
    // but returning a success status is good practice for the script's own logs.
    return ContentService.createTextOutput(JSON.stringify({"status": "success"}))
      .setMimeType(ContentService.MimeType.JSON);
      
  } catch(error) {
    return ContentService.createTextOutput(JSON.stringify({"status": "error", "message": error.message}))
      .setMimeType(ContentService.MimeType.JSON);
  }
}
```

4. Click the **Save** icon (disk).
5. Click **Deploy > New deployment**.
6. Set the "Select type" gear icon to **Web app**.
7. Set "Execute as" to **Me**.
8. Set "Who has access" to **Anyone** (This is critical for the `fetch` POST from GitHub Pages to work anonymously without authentication).
9. Click **Deploy**. Google may prompt you to authorize permissions—allow them.
10. Copy the **Web app URL**. This is your `WEBHOOK_URL`.

## 3. Configuring the Webhook for your Subjects

Because we want the data submission to be seamless, the Web App URL generated in Step 2.10 has already been **hardcoded directly into your experiment's source code**. 

You do **not** need to ask subjects to enter any URL, nor do you need to open Developer Tools to configure their browser.

When the subject finishes the experiment and clicks **Submit All Data**, the JSON payload will be silently sent straight to your Google Sheet!

## 4. Retrieving Data
Your data will appear simultaneously in your Google Spreadsheet.
- Data from the DSB experiment goes into the `DSB` tab.
- Data from the FIP experiment goes into the `FIP` tab.
- Data from the TPB experiment goes into the `TPB` tab.

The actual JSON log will be stored in the third column. You can then download the sheet as CSV and use Python or R to parse the JSON column for your analysis.
