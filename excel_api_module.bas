Attribute VB_Name = "ApiPredictionModule"
Option Explicit

' Excel integration for the Flask endpoint:
'   GET /predict-day-excel?date=YYYY-MM-DD&format=csv
' Writes two columns: hour | predicted_consumption

Private Const DEFAULT_API_BASE As String = "http://localhost:5000"

Public Sub FillPredictionsForDate()
    ' Default wiring:
    ' - Input date in B1 (YYYY-MM-DD)
    ' - Output starts at A4
    FillPredictionsFromCells "B1", "A4", DEFAULT_API_BASE
End Sub

Public Sub FillPredictionsFromCells(ByVal dateCellAddress As String, _
                                    ByVal outputStartCellAddress As String, _
                                    Optional ByVal apiBaseUrl As String = DEFAULT_API_BASE)
    Dim ws As Worksheet
    Dim dateValue As String
    Dim parsedDate As Date
    Dim ymd As String

    Set ws = ActiveSheet
    dateValue = Trim$(CStr(ws.Range(dateCellAddress).Value))

    If Len(dateValue) = 0 Then
        MsgBox "Please input a date in cell " & dateCellAddress & ".", vbExclamation
        Exit Sub
    End If

    On Error GoTo DateParseError
    parsedDate = CDate(dateValue)
    ymd = Format$(parsedDate, "yyyy-mm-dd")
    On Error GoTo 0

    Dim url As String
    url = apiBaseUrl & "/predict-day-excel?date=" & ymd & "&format=csv"

    Dim csvText As String
    csvText = HttpGetText(url)

    WriteCsvHourPrediction ws, outputStartCellAddress, csvText
    MsgBox "Loaded 24 hourly predictions for " & ymd & ".", vbInformation
    Exit Sub

DateParseError:
    MsgBox "Invalid date in " & dateCellAddress & ". Use a valid Excel date or YYYY-MM-DD.", vbCritical
End Sub

Private Function HttpGetText(ByVal url As String) As String
    Dim http As Object
    Set http = CreateObject("MSXML2.XMLHTTP")

    http.Open "GET", url, False
    http.setRequestHeader "Accept", "text/csv"
    http.send

    If http.Status < 200 Or http.Status >= 300 Then
        Err.Raise vbObjectError + 513, "HttpGetText", _
            "HTTP " & http.Status & ": " & http.statusText & vbCrLf & CStr(http.responseText)
    End If

    HttpGetText = CStr(http.responseText)
End Function

Private Sub WriteCsvHourPrediction(ByVal ws As Worksheet, _
                                   ByVal outputStartCellAddress As String, _
                                   ByVal csvText As String)
    Dim startCell As Range
    Dim rows As Variant
    Dim line As Variant
    Dim parts As Variant
    Dim r As Long

    Set startCell = ws.Range(outputStartCellAddress)

    ' Clear a safe area: header + 24 rows, 2 columns
    ws.Range(startCell, startCell.Offset(25, 1)).ClearContents

    rows = Split(Replace(csvText, vbCrLf, vbLf), vbLf)

    ' Header
    startCell.Value = "hour"
    startCell.Offset(0, 1).Value = "predicted_consumption"

    r = 1
    For Each line In rows
        line = Trim$(CStr(line))
        If Len(line) = 0 Then
            GoTo ContinueLoop
        End If

        ' Skip CSV header from API
        If LCase$(line) = "hour,predicted_consumption" Then
            GoTo ContinueLoop
        End If

        parts = Split(CStr(line), ",")
        If UBound(parts) >= 1 Then
            startCell.Offset(r, 0).Value = parts(0)
            startCell.Offset(r, 1).Value = CDbl(parts(1))
            r = r + 1
        End If
ContinueLoop:
    Next line

    ' Optional formatting
    ws.Range(startCell, startCell.Offset(0, 1)).Font.Bold = True
    ws.Range(startCell.Offset(1, 1), startCell.Offset(Application.Max(r - 1, 1), 1)).NumberFormat = "0.00"
End Sub

Public Sub FillPredictionsPromptDate()
    ' Optional utility if you want a prompt instead of reading a cell.
    Dim s As String
    Dim dt As Date
    Dim ymd As String

    s = InputBox("Enter date (YYYY-MM-DD):", "Prediction Date")
    If Len(Trim$(s)) = 0 Then Exit Sub

    On Error GoTo PromptDateError
    dt = CDate(s)
    ymd = Format$(dt, "yyyy-mm-dd")
    On Error GoTo 0

    ActiveSheet.Range("B1").Value = ymd
    FillPredictionsFromCells "B1", "A4", DEFAULT_API_BASE
    Exit Sub

PromptDateError:
    MsgBox "Invalid date format.", vbCritical
End Sub
