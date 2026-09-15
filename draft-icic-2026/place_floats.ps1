[CmdletBinding()]
param([Parameter(Mandatory = $true)][string]$Docx)

# Moves the floating text boxes written by tex_to_docx.py to IEEE positions:
# "Wide" boxes to the top of their page, "Column" boxes to the top of a column,
# stacked when several share one. Page and column are only known after Word
# lays the document out, so this runs inside Word, one pass at a time, until a
# pass reproduces the previous one.
#
# Two feedback loops are broken explicitly:
#  * Column choice. A box at a column top pushes that column's text, which can
#    move its own anchor into the other column. The column chosen for a box is
#    therefore kept while the box stays on the same page, and boxes are placed
#    by an explicit offset from the margin rather than relative to the column.
#  * Page choice. A box can push its anchor onto the next page, where the box
#    no longer fits the earlier page's text, so the anchor returns. When a
#    box's page alternates, its anchor paragraph moves one paragraph later.
#
# Each pass measures every box before moving any: geometry read right after a
# change is transient until Word lays the page out again (a box briefly
# reports a height of 1584 pt). Word returns Single values; they are converted
# to Double because [Math] overloads refuse a Single/Double mix.

$ErrorActionPreference = 'Stop'
$gap = 12.0        # points between a box and the next box below it
$maxPasses = 16
$maxMoves = 6      # anchor moves allowed per box

# Word constants
$wdActiveEndPageNumber = 3
$wdHorizontalPositionRelativeToPage = 5
$wdVerticalPositionRelativeToPage = 6
$relMargin = 0
$relPage = 1
$wdShapeCenter = -999995

function Get-Boxes($doc) {
    @($doc.Shapes |
        Where-Object { $_.Name -like 'Wide *' -or $_.Name -like 'Column *' } |
        Sort-Object { if ($_.Name -like 'Wide *') { 0 } else { 1 } }, { $_.Anchor.Start })
}

function Move-AnchorLater($shape) {
    # The anchor paragraph holds only the box, so copying its FormattedText
    # after the next paragraph carries the box along with it.
    $name = $shape.Name
    $para = $shape.Anchor.Paragraphs.Item(1)
    $after = $para.Next()
    if ($null -eq $after) { return $false }
    $after.Range.InsertParagraphAfter()
    $target = $after.Next()
    $target.Range.FormattedText = $para.Range.FormattedText
    $para.Range.Delete() | Out-Null
    $copies = $target.Range.ShapeRange
    if ($copies.Count -ne 1) { throw "re-anchoring $name left $($copies.Count) boxes in the target paragraph" }
    $copies.Item(1).Name = $name
    return $true
}

$failed = $false
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
try {
    $doc = $word.Documents.Open((Resolve-Path $Docx).Path, $false, $false)
    $setup = $doc.Sections($doc.Sections.Count).PageSetup
    $pageMid = [double]$setup.PageWidth / 2
    $bottomLimit = [double]$setup.PageHeight - [double]$setup.BottomMargin
    $marginTop = [double]$setup.TopMargin
    $rightColLeft = [double]$setup.TextColumns.Width + [double]$setup.TextColumns.Spacing
    # On page 1 the columns start below the title and author band, i.e. at the
    # first paragraph of the last (two-column) section.
    $bodyTop = [double]$doc.Sections($doc.Sections.Count).Range.Paragraphs(1).Range.Information($wdVerticalPositionRelativeToPage)

    $boxCount = (Get-Boxes $doc).Count
    $sticky = @{}
    $history = @{}
    $moves = @{}
    $stable = $false
    $prevSig = ''
    $placed = @()
    for ($pass = 1; $pass -le $maxPasses; $pass++) {
        $doc.Repaginate()
        $boxes = Get-Boxes $doc
        if ($boxes.Count -ne $boxCount) { throw "box count changed from $boxCount to $($boxes.Count)" }

        $items = @(foreach ($s in $boxes) {
            $page = [int]$s.Anchor.Information($wdActiveEndPageNumber)
            $col = 'W'
            if ($s.Name -like 'Column *') {
                $prev = $sticky[$s.Name]
                if ($prev -and $prev.Page -eq $page) {
                    $col = $prev.Col
                } else {
                    $x = [double]$s.Anchor.Information($wdHorizontalPositionRelativeToPage)
                    $col = if ($x -lt $pageMid) { 'L' } else { 'R' }
                }
                $sticky[$s.Name] = @{ Page = $page; Col = $col }
            }
            [pscustomobject]@{
                Shape = $s; Name = $s.Name; Page = $page; Col = $col
                Height = [double]$s.Height; Top = 0.0; Bottom = 0.0
            }
        })

        $next = @{}
        foreach ($it in $items) {
            $pageTop = if ($it.Page -eq 1) { $bodyTop } else { $marginTop }
            foreach ($c in 'L', 'R') {
                if (-not $next.ContainsKey("$($it.Page)$c")) { $next["$($it.Page)$c"] = $pageTop }
            }
            if ($it.Col -eq 'W') {
                $top = [Math]::Max([double]$next["$($it.Page)L"], [double]$next["$($it.Page)R"])
            } else {
                $top = [double]$next["$($it.Page)$($it.Col)"]
            }
            $it.Top = $top
            $it.Bottom = $top + $it.Height
            $end = $it.Bottom + $gap
            if ($it.Col -eq 'W') {
                $next["$($it.Page)L"] = $end
                $next["$($it.Page)R"] = $end
            } else {
                $next["$($it.Page)$($it.Col)"] = $end
            }
        }

        $placed = $items
        $sig = ($items | ForEach-Object { '{0}|{1}|{2}|{3:N0}|{4:N0}' -f $_.Name, $_.Page, $_.Col, $_.Top, $_.Height }) -join ';'
        $stable = $sig -eq $prevSig
        Write-Verbose "pass $pass stable=$stable $sig"
        if ($stable) { break }
        $prevSig = $sig

        $moved = $false
        foreach ($it in $items) {
            if (-not $history.ContainsKey($it.Name)) { $history[$it.Name] = New-Object System.Collections.ArrayList }
            $h = $history[$it.Name]
            [void]$h.Add($it.Page)
            $n = $h.Count
            if ($n -ge 3 -and $h[$n - 1] -eq $h[$n - 3] -and $h[$n - 1] -ne $h[$n - 2]) {
                if ([int]$moves[$it.Name] -ge $maxMoves) { throw "$($it.Name) still alternates pages after $maxMoves anchor moves" }
                if (Move-AnchorLater $it.Shape) {
                    $moves[$it.Name] = [int]$moves[$it.Name] + 1
                    $h.Clear()
                    $sticky.Remove($it.Name)
                    $moved = $true
                    Write-Verbose "pass $pass moved the anchor of $($it.Name) one paragraph later"
                }
            }
        }
        if ($moved) {
            $prevSig = ''
            continue
        }

        foreach ($it in $items) {
            $s = $it.Shape
            $s.RelativeHorizontalPosition = $relMargin
            $s.Left = switch ($it.Col) {
                'W' { [single]$wdShapeCenter }
                'L' { [single]0 }
                'R' { [single]$rightColLeft }
            }
            $s.RelativeVerticalPosition = $relPage
            $s.Top = [single]$it.Top
        }
    }

    foreach ($p in $placed) {
        $flag = ''
        if ($p.Bottom -gt $bottomLimit + 0.5) { $flag = ' OVERFLOWS PAGE'; $failed = $true }
        foreach ($q in $placed) {
            if ($q.Name -ne $p.Name -and $q.Page -eq $p.Page -and
                ($q.Col -eq $p.Col -or $q.Col -eq 'W' -or $p.Col -eq 'W') -and
                $q.Top -lt $p.Bottom -and $p.Top -lt $q.Bottom) {
                $flag += " OVERLAPS $($q.Name)"
                $failed = $true
            }
        }
        $moveNote = if ($moves[$p.Name]) { " (anchor moved $($moves[$p.Name])x)" } else { '' }
        '{0,-10} page {1} col {2} top {3,6:N1} bottom {4,6:N1}{5}{6}' -f $p.Name, $p.Page, $p.Col, $p.Top, $p.Bottom, $moveNote, $flag
    }
    if (-not $stable) {
        "layout did not settle after $maxPasses passes"
        $failed = $true
    }
    $doc.Save()
    $doc.Close($false)
} finally {
    $word.Quit()
}
if ($failed) { exit 1 }
