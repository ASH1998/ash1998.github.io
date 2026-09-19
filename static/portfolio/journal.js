(() => {
  const canvas=document.createElement('canvas');canvas.width=canvas.height=180;
  const ctx=canvas.getContext('2d'),noise=ctx.createImageData(180,180);
  let seed=29;for(let i=0;i<noise.data.length;i+=4){seed=(seed*1664525+1013904223)>>>0;noise.data[i]=noise.data[i+1]=noise.data[i+2]=seed/4294967296*255;noise.data[i+3]=255;}
  ctx.putImageData(noise,0,0);document.querySelector('.grain').style.backgroundImage=`url(${canvas.toDataURL()})`;
})();
