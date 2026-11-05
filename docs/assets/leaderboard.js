(() => {
  'use strict';
  const table=document.querySelector('.leaderboard-table');
  if(!table)return;
  const body=table.tBodies[0], rows=[...body.rows], input=document.getElementById('model-search');
  let sortKey='overall', direction='descending';
  function update(){
    const query=input.value.trim().toLocaleLowerCase();
    const sorted=[...rows].sort((a,b)=>{
      const delta=sortKey==='model'?a.dataset.model.localeCompare(b.dataset.model):Number(a.dataset[sortKey])-Number(b.dataset[sortKey]);
      return (direction==='ascending'?delta:-delta)||Number(a.dataset.rank)-Number(b.dataset.rank);
    });
    let count=0;
    document.getElementById('empty-row')?.remove();
    sorted.forEach(row=>{row.hidden=!row.dataset.model.toLocaleLowerCase().includes(query);if(!row.hidden)count++;body.append(row);});
    if(!count){const row=body.insertRow();row.id='empty-row';const cell=row.insertCell();cell.colSpan=8;cell.className='empty-cell';cell.textContent='No models found. Try a different name.';}
    document.getElementById('visible-count').textContent=`${count} of ${rows.length} models`;
    document.getElementById('filter-status').textContent=`Showing ${count} models. Sorted by ${sortKey.replace(/([A-Z])/g,' $1').toLowerCase()}, ${direction}.`;
    table.querySelectorAll('th[data-sort]').forEach(th=>{
      const active=th.dataset.sort===sortKey;th.setAttribute('aria-sort',active?direction:'none');
      const arrow=th.querySelector('.sort-arrow');if(arrow)arrow.textContent=active?(direction==='descending'?'↓':'↑'):'↕';
    });
  }
  table.querySelectorAll('th[data-sort] button').forEach(button=>button.addEventListener('click',()=>{
    const key=button.closest('th').dataset.sort;
    direction=key===sortKey?(direction==='descending'?'ascending':'descending'):(key==='model'?'ascending':'descending');sortKey=key;update();
  }));
  input.addEventListener('input',update);
  input.addEventListener('keydown',event=>{if(event.key==='Escape'){input.value='';update();}});
  update();
})();
