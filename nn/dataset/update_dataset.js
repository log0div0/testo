
const fs = require('fs')
const path = require('path')

let DATASET_DIR = './homm3'

async function main() {
	const files = await fs.promises.readdir(DATASET_DIR);
	for (const file of files) {
		let parsed = path.parse(file)
		if (parsed.base == 'meta.json') {
			continue
		}
		if (parsed.ext.toLowerCase() != '.json') {
			continue
		}
		let metadata_path = path.join(DATASET_DIR, file)
		let data = await fs.promises.readFile(metadata_path)
		metadata = JSON.parse(data)
		delete metadata.icons
		let objs = {}
		for (let obj_id in metadata.objs) {
			let obj = metadata.objs[obj_id]
			if (obj.tag == 'sholar') {
				obj.tag = 'scholar'
			}
			objs[obj_id] = obj
		}
		metadata.objs = objs
		data = JSON.stringify(metadata, function(key, val) {
			return val.toFixed ? Number(val.toFixed(2)) : val;
		}, '\t')
		await fs.promises.writeFile(metadata_path, data)
	}
}

main()
