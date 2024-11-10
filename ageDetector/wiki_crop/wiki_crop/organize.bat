@echo off
setlocal enabledelayedexpansion

rem Create the "processed" folder if it doesn't exist
md processed 2>nul

rem Loop through all files with the specified pattern in all subfolders
for /r %%F in (*_*_*.jpg) do (
    rem Get the size of the file
    for %%A in ("%%F") do (
        set "file_size=%%~zA"
        rem Check if the file size is greater than or equal to 15KB
        if !file_size! GEQ 15360 (
            rem Extract the date of birth from the file name
            for /f "tokens=2 delims=_" %%B in ("%%~nF") do (
                set "dob=%%B"
                rem Extract the photo date from the file name
                for /f "tokens=3 delims=_" %%C in ("%%~nF") do (
                    set "photo_date=%%C"
                    rem Calculate the age at the time of the photo
                    set /a "age=photo_date-dob"
                    rem Check if the age is greater than or equal to zero
                    if !age! GEQ 0 (
                        rem Create a subfolder with the calculated age under "processed"
                        md "processed\!age!" 2>nul
                        rem Move the image file to the processed folder
                        xcopy "%%F" "processed\!age!" /y >nul
                    ) else (
                        echo Ignoring file %%F because the calculated age is negative.
                    )
                )
            )
        ) else (
            echo Ignoring file %%F because it is less than 15KB in size.
        )
    )
)

endlocal
