@echo off
setlocal enabledelayedexpansion
set count=0
for /f %%i in ('dir /b /a-d *.jpg ^| find "" /v /c') do set /a count=%%i

set /a cut_count=count/10

if not exist "valid" mkdir "valid"

set moved_count=0
for /f "delims=" %%f in ('dir /b /a-d *.jpg') do (
    set /a "rand=!random! %% 10"
    if !rand! equ 0 (
        move "%%f" "valid"
        set /a moved_count+=1
    )
    if !moved_count! geq %cut_count% (
        exit /b
    )
)
