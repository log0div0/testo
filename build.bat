cmake -G "Visual Studio 15 2017 Win64" %~dp0
msbuild Project.sln /p:Platform=x64 /p:Configuration=Release
