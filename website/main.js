
import express from 'express'
import assert from 'assert'

const app = express()
const port = 80

app.set('views', __dirname + '/views');
app.set('view engine', 'js');
app.engine('js', require('express-react-views').createEngine());
app.use(express.static('public'))

app.get('/', async function(req, res) {
	res.render('index')
})

app.listen(port, () => console.log(`Listening on port ${port}`))
