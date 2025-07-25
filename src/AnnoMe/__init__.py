## Fix for PosixPath problem on Windows
# thanks to https://stackoverflow.com/a/64393589
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath