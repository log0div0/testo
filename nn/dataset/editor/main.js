
const { app, BrowserWindow } = require('electron')

function createWindow () {
	const win = new BrowserWindow({
		webPreferences: {
			nodeIntegration: true
		}
	})

	win.loadFile('index.html')
	win.maximize()
}

app.whenReady().then(createWindow)
