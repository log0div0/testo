
import path from 'path'

export let families = [
	'Arial',
	'Calibri',
	'Consolas',
	'Courier New',
	'DejaVu Sans Condensed',
	'DejaVu Serif Condensed',
	'Georgia',
	'Lucida Console',
	'Microsoft Sans Serif',
	'Segoe UI',
	'Tahoma',
	'Times New Roman',
	'Trebuchet MS',
	'Ubuntu',
	'Verdana'
]
export let weights = [
	'normal',
	'bold'
]
export let styles = [
	'normal',
	'italic'
]

export let fontsDir = 'file://' + path.join(__dirname, 'truetype')

export let css = `
@font-face {
	font-family: 'Arial';
	font-weight: normal;
	font-style: normal;
	src: url('${fontsDir + '/arial/arial.ttf'}') format('truetype');
}

@font-face {
	font-family: 'Arial';
	font-weight: bold;
	font-style: normal;
	src: url('${fontsDir + '/arial/arialbd.ttf'}') format('truetype');
}

@font-face {
	font-family: 'Arial';
	font-weight: normal;
	font-style: italic;
	src: url('${fontsDir + '/arial/ariali.ttf'}') format('truetype');
}

@font-face {
	font-family: 'Arial';
	font-weight: bold;
	font-style: italic;
	src: url('${fontsDir + '/arial/arialbi.ttf'}') format('truetype');
}

@font-face {
	font-family: 'Calibri';
	font-weight: normal;
	font-style: normal;
	src: url('${fontsDir + '/calibri/calibri.ttf'}') format('truetype');
}

@font-face {
	font-family: 'Calibri';
	font-weight: bold;
	font-style: normal;
	src: url('${fontsDir + '/calibri/calibrib.ttf'}') format('truetype');
}

@font-face {
	font-family: 'Calibri';
	font-weight: normal;
	font-style: italic;
	src: url('${fontsDir + '/calibri/calibrii.ttf'}') format('truetype');
}

@font-face {
	font-family: 'Calibri';
	font-weight: bold;
	font-style: italic;
	src: url('${fontsDir + '/calibri/calibriz.ttf'}') format('truetype');
}

@font-face {
	font-family: 'Consolas';
	font-weight: normal;
	font-style: normal;
	src: url('${fontsDir + '/consolas/consola.ttf'}') format('truetype');
}

@font-face {
	font-family: 'Consolas';
	font-weight: bold;
	font-style: normal;
	src: url('${fontsDir + '/consolas/consolab.ttf'}') format('truetype');
}

@font-face {
	font-family: 'Consolas';
	font-weight: normal;
	font-style: italic;
	src: url('${fontsDir + '/consolas/consolai.ttf'}') format('truetype');
}

@font-face {
	font-family: 'Consolas';
	font-weight: bold;
	font-style: italic;
	src: url('${fontsDir + '/consolas/consolaz.ttf'}') format('truetype');
}

@font-face {
	font-family: 'Courier New';
	font-weight: normal;
	font-style: normal;
	src: url('${fontsDir + '/courier_new/cour.ttf'}') format('truetype');
}

@font-face {
	font-family: 'Courier New';
	font-weight: bold;
	font-style: normal;
	src: url('${fontsDir + '/courier_new/courbd.ttf'}') format('truetype');
}

@font-face {
	font-family: 'Courier New';
	font-weight: normal;
	font-style: italic;
	src: url('${fontsDir + '/courier_new/couri.ttf'}') format('truetype');
}

@font-face {
	font-family: 'Courier New';
	font-weight: bold;
	font-style: italic;
	src: url('${fontsDir + '/courier_new/courbi.ttf'}') format('truetype');
}

@font-face {
	font-family: 'DejaVu Sans Condensed';
	font-weight: normal;
	font-style: normal;
	src: url('${fontsDir + '/dejavu/DejaVuSansCondensed.ttf'}') format('truetype');
}

@font-face {
	font-family: 'DejaVu Sans Condensed';
	font-weight: bold;
	font-style: normal;
	src: url('${fontsDir + '/dejavu/DejaVuSansCondensed-Bold.ttf'}') format('truetype');
}

@font-face {
	font-family: 'DejaVu Sans Condensed';
	font-weight: normal;
	font-style: italic;
	src: url('${fontsDir + '/dejavu/DejaVuSansCondensed-Oblique.ttf'}') format('truetype');
}

@font-face {
	font-family: 'DejaVu Sans Condensed';
	font-weight: bold;
	font-style: italic;
	src: url('${fontsDir + '/dejavu/DejaVuSansCondensed-BoldOblique.ttf'}') format('truetype');
}

@font-face {
	font-family: 'DejaVu Serif Condensed';
	font-weight: normal;
	font-style: normal;
	src: url('${fontsDir + '/dejavu/DejaVuSerifCondensed.ttf'}') format('truetype');
}

@font-face {
	font-family: 'DejaVu Serif Condensed';
	font-weight: bold;
	font-style: normal;
	src: url('${fontsDir + '/dejavu/DejaVuSerifCondensed-Bold.ttf'}') format('truetype');
}

@font-face {
	font-family: 'DejaVu Serif Condensed';
	font-weight: normal;
	font-style: italic;
	src: url('${fontsDir + '/dejavu/DejaVuSerifCondensed-Italic.ttf'}') format('truetype');
}

@font-face {
	font-family: 'DejaVu Serif Condensed';
	font-weight: bold;
	font-style: italic;
	src: url('${fontsDir + '/dejavu/DejaVuSerifCondensed-BoldItalic.ttf'}') format('truetype');
}

@font-face {
	font-family: 'Georgia';
	font-weight: normal;
	font-style: normal;
	src: url('${fontsDir + '/georgia/georgia.ttf'}') format('truetype');
}

@font-face {
	font-family: 'Georgia';
	font-weight: bold;
	font-style: normal;
	src: url('${fontsDir + '/georgia/georgiab.ttf'}') format('truetype');
}

@font-face {
	font-family: 'Georgia';
	font-weight: normal;
	font-style: italic;
	src: url('${fontsDir + '/georgia/georgiai.ttf'}') format('truetype');
}

@font-face {
	font-family: 'Georgia';
	font-weight: bold;
	font-style: italic;
	src: url('${fontsDir + '/georgia/georgiaz.ttf'}') format('truetype');
}

@font-face {
	font-family: 'Lucida Console';
	font-weight: normal;
	font-style: normal;
	src: url('${fontsDir + '/lucida_console/lucon.ttf'}') format('truetype');
}

@font-face {
	font-family: 'Microsoft Sans Serif';
	font-weight: normal;
	font-style: normal;
	src: url('${fontsDir + '/microsoft_sans_serif/micross.ttf'}') format('truetype');
}

@font-face {
	font-family: 'Segoe UI';
	font-weight: normal;
	font-style: normal;
	src: url('${fontsDir + '/segoe_ui/segoeui.ttf'}') format('truetype');
}

@font-face {
	font-family: 'Segoe UI';
	font-weight: bold;
	font-style: normal;
	src: url('${fontsDir + '/segoe_ui/segoeuib.ttf'}') format('truetype');
}

@font-face {
	font-family: 'Segoe UI';
	font-weight: normal;
	font-style: italic;
	src: url('${fontsDir + '/segoe_ui/segoeuii.ttf'}') format('truetype');
}

@font-face {
	font-family: 'Segoe UI';
	font-weight: bold;
	font-style: italic;
	src: url('${fontsDir + '/segoe_ui/segoeuiz.ttf'}') format('truetype');
}

@font-face {
	font-family: 'Tahoma';
	font-weight: normal;
	font-style: normal;
	src: url('${fontsDir + '/tahoma/tahoma.ttf'}') format('truetype');
}

@font-face {
	font-family: 'Tahoma';
	font-weight: bold;
	font-style: normal;
	src: url('${fontsDir + '/tahoma/tahomabd.ttf'}') format('truetype');
}

@font-face {
	font-family: 'Times New Roman';
	font-weight: normal;
	font-style: normal;
	src: url('${fontsDir + '/times_new_roman/times.ttf'}') format('truetype');
}

@font-face {
	font-family: 'Times New Roman';
	font-weight: bold;
	font-style: normal;
	src: url('${fontsDir + '/times_new_roman/timesbd.ttf'}') format('truetype');
}

@font-face {
	font-family: 'Times New Roman';
	font-weight: normal;
	font-style: italic;
	src: url('${fontsDir + '/times_new_roman/timesi.ttf'}') format('truetype');
}

@font-face {
	font-family: 'Times New Roman';
	font-weight: bold;
	font-style: italic;
	src: url('${fontsDir + '/times_new_roman/timesbi.ttf'}') format('truetype');
}

@font-face {
	font-family: 'Trebuchet MS';
	font-weight: normal;
	font-style: normal;
	src: url('${fontsDir + '/trebuchet_ms/trebuc.ttf'}') format('truetype');
}

@font-face {
	font-family: 'Trebuchet MS';
	font-weight: bold;
	font-style: normal;
	src: url('${fontsDir + '/trebuchet_ms/trebucbd.ttf'}') format('truetype');
}

@font-face {
	font-family: 'Trebuchet MS';
	font-weight: normal;
	font-style: italic;
	src: url('${fontsDir + '/trebuchet_ms/trebucit.ttf'}') format('truetype');
}

@font-face {
	font-family: 'Trebuchet MS';
	font-weight: bold;
	font-style: italic;
	src: url('${fontsDir + '/trebuchet_ms/trebucbi.ttf'}') format('truetype');
}

@font-face {
	font-family: 'Ubuntu';
	font-weight: normal;
	font-style: normal;
	src: url('${fontsDir + '/ubuntu/Ubuntu-R.ttf'}') format('truetype');
}

@font-face {
	font-family: 'Ubuntu';
	font-weight: bold;
	font-style: normal;
	src: url('${fontsDir + '/ubuntu/Ubuntu-B.ttf'}') format('truetype');
}

@font-face {
	font-family: 'Ubuntu';
	font-weight: normal;
	font-style: italic;
	src: url('${fontsDir + '/ubuntu/Ubuntu-RI.ttf'}') format('truetype');
}

@font-face {
	font-family: 'Ubuntu';
	font-weight: bold;
	font-style: italic;
	src: url('${fontsDir + '/ubuntu/Ubuntu-BI.ttf'}') format('truetype');
}

@font-face {
	font-family: 'Verdana';
	font-weight: normal;
	font-style: normal;
	src: url('${fontsDir + '/verdana/verdana.ttf'}') format('truetype');
}

@font-face {
	font-family: 'Verdana';
	font-weight: bold;
	font-style: normal;
	src: url('${fontsDir + '/verdana/verdanab.ttf'}') format('truetype');
}

@font-face {
	font-family: 'Verdana';
	font-weight: normal;
	font-style: italic;
	src: url('${fontsDir + '/verdana/verdanai.ttf'}') format('truetype');
}

@font-face {
	font-family: 'Verdana';
	font-weight: bold;
	font-style: italic;
	src: url('${fontsDir + '/verdana/verdanaz.ttf'}') format('truetype');
}
`
