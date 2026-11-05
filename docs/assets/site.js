document.documentElement.classList.add('js');

(() => {
  'use strict';
  const video=document.getElementById('demo-video'), playButton=document.getElementById('video-play');
  if(video&&playButton){
    playButton.addEventListener('click',async()=>{try{await video.play();playButton.hidden=true;}catch{playButton.hidden=true;video.focus();}});
    video.addEventListener('play',()=>{playButton.hidden=true;});
    video.addEventListener('ended',()=>{playButton.hidden=false;});
  }
  const copy=document.getElementById('copy-citation');
  if(copy)copy.addEventListener('click',async()=>{
    const text=document.getElementById('citation-text').textContent;
    try{await navigator.clipboard.writeText(text);}catch{
      const area=document.createElement('textarea');area.value=text;area.style.position='fixed';area.style.opacity='0';document.body.append(area);area.select();const ok=document.execCommand('copy');area.remove();if(!ok){document.getElementById('copy-status').textContent='Please select and copy the citation below.';return;}
    }
    copy.textContent='Copied ✓';document.getElementById('copy-status').textContent='Citation copied to clipboard.';
    setTimeout(()=>{copy.textContent='Copy citation ⧉';},2200);
  });
})();
