Option Explicit

' VBScript that fetches hourly predictions from the API
' and writes them into Excel (A4:B... on active sheet).

Const API_BASE = "http://localhost:5000"
Const START_CELL_ROW = 4
Const START_CELL_COL = 1 ' A

Dim requestedDate
requestedDate = InputBox("Enter prediction date (YYYY-MM-DD):", "Prediction Date", "2018-08-04")
If Trim(requestedDate) = "" Then
    WScript.Quit 0
End If

If Not IsDate(requestedDate) Then
    MsgBox "Invalid date. Use YYYY-MM-DD.", vbCritical, "Error"
    WScript.Quit 1
End If

Dim dateValue
dateValue = CDate(requestedDate)
Dim ymd
ymd = Year(dateValue) & "-" & Right("0" & Month(dateValue), 2) & "-" & Right("0" & Day(dateValue), 2)

Dim url
url = API_BASE & "/predict-day-excel?date=" & ymd & "&format=csv"

Dim csvText
csvText = HttpGet(url)

Dim xl, wb, ws
Set xl = GetExcelApp()
If xl Is Nothing Then
    MsgBox "Could not start or connect to Excel.", vbCritical, "Error"
    WScript.Quit 1
End If

If xl.Workbooks.Count = 0 Then
    Set wb = xl.Workbooks.Add
Else
    Set wb = xl.ActiveWorkbook
End If

Set ws = wb.ActiveSheet
WriteCsvToSheet ws, csvText

xl.Visible = True
MsgBox "Loaded 24 hourly predictions for " & ymd & " into " & ws.Name & " (A4:B).", vbInformation, "Done"

Function GetExcelApp()
    On Error Resume Next

    Dim app
    Set app = GetObject(, "Excel.Application")
    If app Is Nothing Then
        Set app = CreateObject("Excel.Application")
    End If

    On Error GoTo 0
    Set GetExcelApp = app
End Function

Function HttpGet(ByVal endpoint)
    Dim http
    Set http = CreateObject("MSXML2.XMLHTTP")

    http.Open "GET", endpoint, False
    http.setRequestHeader "Accept", "text/csv"
    http.send

    If http.Status < 200 Or http.Status >= 300 Then
        MsgBox "API request failed: HTTP " & http.Status & vbCrLf & http.responseText, vbCritical, "API Error"
        WScript.Quit 1
    End If

    HttpGet = CStr(http.responseText)
End Function

Sub WriteCsvToSheet(ByVal ws, ByVal csv)
    Dim normalized, lines, i, line, parts, rowIndex

    normalized = Replace(csv, vbCrLf, vbLf)
    lines = Split(normalized, vbLf)

    ' Clear output area: header + 24 rows, 2 cols
    ws.Range(ws.Cells(START_CELL_ROW, START_CELL_COL), ws.Cells(START_CELL_ROW + 25, START_CELL_COL + 1)).ClearContents

    ws.Cells(START_CELL_ROW, START_CELL_COL).Value = "hour"
    ws.Cells(START_CELL_ROW, START_CELL_COL + 1).Value = "predicted_consumption"
    ws.Range(ws.Cells(START_CELL_ROW, START_CELL_COL), ws.Cells(START_CELL_ROW, START_CELL_COL + 1)).Font.Bold = True

    rowIndex = START_CELL_ROW + 1
    For i = 0 To UBound(lines)
        line = Trim(CStr(lines(i)))
        If line <> "" Then
            If LCase(line) <> "hour,predicted_consumption" Then
                parts = Split(line, ",")
                If UBound(parts) >= 1 Then
                    ws.Cells(rowIndex, START_CELL_COL).Value = parts(0)
                    ws.Cells(rowIndex, START_CELL_COL + 1).Value = CDbl(parts(1))
                    rowIndex = rowIndex + 1
                End If
            End If
        End If
    Next

    ws.Range(ws.Cells(START_CELL_ROW + 1, START_CELL_COL + 1), ws.Cells(START_CELL_ROW + 24, START_CELL_COL + 1)).NumberFormat = "0.00"
End Sub
