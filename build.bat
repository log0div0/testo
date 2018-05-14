cmake -G "Visual Studio 15 2017 Win64" %~dp0
msbuild testo.sln /p:Platform=x64 /p:Configuration=Release
