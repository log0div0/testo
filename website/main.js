
import express from 'express'
import assert from 'assert'
import sass from 'node-sass'

const app = express()
const port = 80

app.set('views', __dirname + '/views');
app.set('view engine', 'js');
app.engine('js', require('express-react-views').createEngine());

app.get('/', (req, res) => {
	res.render('home')
})

app.get('/main.css', (req, res) => {
	sass.render({file: 'main.scss'}, function(err, result) {
		if (err) {
			return res.status(500).send(err.message)
		}
		res.set('Content-Type', 'text/css')
		res.send(result.css)
	})
})

app.get('/docs/*', require('./docs'))

app.use(express.static('public'))
app.listen(port, () => console.log(`Listening on port ${port}`))
