
import React from 'react'
import { useSelector, useDispatch } from 'react-redux'
import path from 'path'

function DocCard({id}) {
	let dispatch = useDispatch()
	let openDoc = () => dispatch({
		type: "OPEN_DOC",
		id
	})
	return (
		<div className="doc-card">
			<style jsx>{`
				.doc-card {
					border: 1px solid gray;
					padding: 20px;
					border-radius: 10px;
					background-color: white;
				}
				button {
					margin-top: 15px;
				}
				img {
					width: 100%;
				}
			`}</style>
			<img src={path.join(DATASET_DIR, id + '.png')}></img>
			<button onClick={openDoc}>{`${id}.png`}</button>
		</div>
	)
}

function DocList() {
	let docs_ids = useSelector(state => state.docs_ids)
	let filters = useSelector(state => state.filters)
	let dispatch = useDispatch()

	let setFilter = event => {
		let target = event.target
		let filterName = target.id
		let filterValue = target.type === 'checkbox' ? target.checked : target.value;
		dispatch({
			type: "SET_FILTER",
			name: filterName,
			value: filterValue
		})
	}

	return (
		<>
			<style jsx global>{`
				body {
					margin: 0;
					padding: 0;
					background-color: #F0F0F0;
				}
			`}</style>
			<style jsx>{`
				.container {
					width: 1200px;
					margin: 20px auto;
				}
				.doc-filters {
					padding: 20px;
					border: 1px solid gray;
					border-radius: 10px;
					margin-bottom: 20px;
				}
				.doc-grid {
					display: grid;
					grid-template-columns: repeat(3, 1fr);
					grid-column-gap: 25px;
					grid-row-gap: 25px;
				}
			`}</style>
			<div className="container">
				<div className="doc-filters">
					<input type="checkbox" id="unverified_only" checked={filters.unverified_only} onChange={setFilter}/>
					<label htmlFor="unverified_only">Unverified Only</label>
				</div>
				<div className="doc-grid">
					{
						docs_ids.map(id => <DocCard key={id} id={id}/>)
					}
				</div>
			</div>
		</>
	)
}

export default DocList
