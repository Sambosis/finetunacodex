param([Parameter(Mandatory=$true)][string]$InputPath, [Parameter(Mandatory=$true)][string]$OutputPath)
# Parse untrusted examples as text. Never invoke them.
$ErrorActionPreference = 'Stop'
$results = foreach ($line in [IO.File]::ReadLines($InputPath)) {
    $row = $line | ConvertFrom-Json
    $tokens = $null
    $parseErrors = $null
    $ast = [System.Management.Automation.Language.Parser]::ParseInput($row.command, [ref]$tokens, [ref]$parseErrors)
    $commands = @($ast.FindAll({param($n) $n -is [System.Management.Automation.Language.CommandAst]}, $true) | ForEach-Object {
        @{name=$_.GetCommandName(); parameters=@($_.CommandElements | Where-Object {$_ -is [System.Management.Automation.Language.CommandParameterAst]} | ForEach-Object {$_.ParameterName})}
    })
    $defined = @($ast.FindAll({param($n) $n -is [System.Management.Automation.Language.AssignmentStatementAst]}, $true) | ForEach-Object {
        $_.Left.FindAll({param($n) $n -is [System.Management.Automation.Language.VariableExpressionAst]}, $true) | ForEach-Object {$_.VariablePath.UserPath}
    })
    $defined += @($ast.FindAll({param($n) $n -is [System.Management.Automation.Language.ForEachStatementAst]}, $true) | ForEach-Object {$_.Variable.VariablePath.UserPath})
    $defined += @($ast.FindAll({param($n) $n -is [System.Management.Automation.Language.ParameterAst]}, $true) | ForEach-Object {$_.Name.VariablePath.UserPath})
    $used = @($ast.FindAll({param($n) $n -is [System.Management.Automation.Language.VariableExpressionAst]}, $true) | ForEach-Object {$_.VariablePath.UserPath} | Sort-Object -Unique)
    @{id=$row.id; errors=@($parseErrors | ForEach-Object {$_.ErrorId}); commands=$commands; defined=$defined; variables=$used; parser_version=$PSVersionTable.PSVersion.ToString()} | ConvertTo-Json -Depth 10 -Compress
}
[IO.File]::WriteAllLines($OutputPath, [string[]]$results, (New-Object Text.UTF8Encoding($false)))
