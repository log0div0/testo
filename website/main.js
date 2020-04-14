
import express from 'express'
import assert from 'assert'

const app = express()
const port = 80

app.set('views', __dirname + '/views');
app.set('view engine', 'js');
app.engine('js', require('express-react-views').createEngine());
app.use(express.static('public'))

app.get('/', require('./routes/home'))
app.get('/docs/*', require('./routes/docs'))

app.listen(port, () => console.log(`Listening on port ${port}`))
