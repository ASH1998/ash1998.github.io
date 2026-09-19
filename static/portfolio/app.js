(() => {
  'use strict';
  document.querySelector('.hero-copy').after(document.querySelector('.particle-stage'));
  let seed=29;
  const rand=()=>{seed=(seed*1664525+1013904223)>>>0;return seed/4294967296;};
  const clamp=(x,a=0,b=1)=>Math.max(a,Math.min(b,x));
  const smooth=t=>{t=clamp(t);return t*t*t*(t*(t*6-15)+10);};
  const mix=(a,b,t)=>a+(b-a)*t,tau=Math.PI*2;
  const grain=document.createElement('canvas');grain.width=grain.height=180;
  const gc=grain.getContext('2d'),noise=gc.createImageData(180,180);
  for(let i=0;i<noise.data.length;i+=4){const v=rand()*255;noise.data[i]=noise.data[i+1]=noise.data[i+2]=v;noise.data[i+3]=255;}
  gc.putImageData(noise,0,0);document.querySelector('.grain').style.backgroundImage=`url(${grain.toDataURL()})`;
  const canvas=document.querySelector('#particles'),ctx=canvas.getContext('2d');
  const sculpture=document.querySelector('#sculpture'),label=document.querySelector('#shape-name'),motion=document.querySelector('#motion');
  const hint=document.querySelector('.interaction-hint');
  const reduced=matchMedia('(prefers-reduced-motion: reduce)'),mobile=matchMedia('(max-width:700px)');
  const count=mobile.matches?3456:7776;
  const names=['Sphere','Cube','Octahedron','Orbit','Catenoid','Möbius','Black hole','Double helix','Trefoil','Gyroscope'];
  const quirks=['Click to disperse','Click to spin faster','Click to send a ripple','Click to stir the orbit','Click to flex the surface','Click to twist the ribbon','Click to pull spacetime inward','Click to unzip the strands','Click to send a knot pulse','Click to accelerate the rings'];
  const forms=names.map(()=>new Float32Array(count*6));
  const current=new Float32Array(count*6),origin=new Float32Array(count*6);
  const seeds=new Float32Array(count*4),rendered=new Float32Array(count*5),buckets=Array.from({length:40},()=>[]);
  function surface(form,i,x,y,z,nx,ny,nz){const d=Math.hypot(nx,ny,nz)||1;forms[form].set([x,y,z,nx/d,ny/d,nz/d],i*6);}
  function mobius(u,v){return [(.79+v*Math.cos(u/2))*Math.cos(u),v*Math.sin(u/2),(.79+v*Math.cos(u/2))*Math.sin(u)];}
  function trefoil(t){const r=.67+.24*Math.cos(3*t);return [r*Math.cos(2*t),.27*Math.sin(3*t),r*Math.sin(2*t)];}
  for(let i=0;i<count;i++){
    const u=(i*.618033988749895)%1,v=(i+.5)/count,z=1-2*v,r=Math.sqrt(1-z*z),a=tau*u;
    const x=r*Math.cos(a),y=z,zz=r*Math.sin(a);
    surface(0,i,x,y,zz,x,y,zz);
    // Surface normals preserve coherent face highlights as the light follows the pointer.
    const face=i%6,grid=Math.ceil(Math.sqrt(count/6)),cell=Math.floor(i/6);
    const g=((cell%grid+.5+(rand()-.5)*.34)/grid*2-1)*.75;
    const h=((Math.floor(cell/grid)+.5+(rand()-.5)*.34)/grid*2-1)*.75;
    const axis=Math.floor(face/2),sign=face%2?1:-1,p=[g,h,0],n=[0,0,0];
    if(axis===0){p[0]=sign*.75;p[1]=g;p[2]=h;}else if(axis===1){p[0]=g;p[1]=sign*.75;p[2]=h;}else{p[0]=g;p[1]=h;p[2]=sign*.75;}n[axis]=sign;
    surface(1,i,...p,...n);
    const sum=Math.abs(x)+Math.abs(y)+Math.abs(zz);
    surface(2,i,x/sum*1.3,y/sum*1.3,zz/sum*1.3,Math.sign(x),Math.sign(y),Math.sign(zz));
    const b=tau*v,rr=.77+.28*Math.cos(b);
    surface(3,i,rr*Math.cos(a),.28*Math.sin(b),rr*Math.sin(a),Math.cos(a)*Math.cos(b),Math.sin(b),Math.sin(a)*Math.cos(b));
    const cy=(v*2-1)*.96,cr=.42*Math.cosh(cy*1.32);
    surface(4,i,cr*Math.cos(a),cy,cr*Math.sin(a),Math.cos(a),-.42*1.32*Math.sinh(cy*1.32),Math.sin(a));
    const mv=(v*2-1)*.3,mp=mobius(a,mv),mu=mobius(a+.001,mv),mw=mobius(a,mv+.001);
    const du=mu.map((e,k)=>e-mp[k]),dv=mw.map((e,k)=>e-mp[k]);
    surface(5,i,...mp,du[1]*dv[2]-du[2]*dv[1],du[2]*dv[0]-du[0]*dv[2],du[0]*dv[1]-du[1]*dv[0]);
    // An annular surface leaves a real central void, with curved particle filaments.
    const br=.38+.77*(Math.floor(i/72)+.5)/Math.ceil(count/72);
    const ba=(i%72)/72*tau+4.8*Math.pow(1-br/1.16,1.55)+(rand()-.5)*.009,bz=.18*Math.sin(2*ba)*(br-.38);
    surface(6,i,br*Math.cos(ba),br*Math.sin(ba),bz,0,0,1);
    const helixSide=i%2?1:-1,ha=v*tau*2.1+(helixSide<0?Math.PI:0),hb=a,hy=v*1.95-.975;
    if(i%5===0){const rung=Math.floor(v*27)/27,ra=rung*tau*2.1,along=u*2-1;surface(7,i,along*.5*Math.cos(ra),(rung*1.95-.975)+Math.cos(a)*.022,along*.5*Math.sin(ra)+Math.sin(a)*.022,Math.cos(a),Math.sin(a),0);}
    else surface(7,i,(.5+.065*Math.cos(hb))*Math.cos(ha),hy+.065*Math.sin(hb),(.5+.065*Math.cos(hb))*Math.sin(ha),Math.cos(hb)*Math.cos(ha),Math.sin(hb),Math.cos(hb)*Math.sin(ha));
    const tc=trefoil(a),tn=trefoil(a+.001),td=tn.map((n,k)=>n-tc[k]),tdl=Math.hypot(...td);for(let k=0;k<3;k++)td[k]/=tdl;
    const normalLength=Math.hypot(td[0],td[2]),normal=[td[2]/normalLength,0,-td[0]/normalLength];
    const binormal=[td[1]*normal[2],td[2]*normal[0]-td[0]*normal[2],-td[1]*normal[0]];
    const tube=normal.map((n,k)=>n*Math.cos(b)+binormal[k]*Math.sin(b));
    surface(8,i,...tc.map((n,k)=>n+tube[k]*.105),...tube);
    const ring=i%3,gr=.88+.047*Math.cos(b),gp=[gr*Math.cos(a),gr*Math.sin(a),.047*Math.sin(b)],gn=[Math.cos(a)*Math.cos(b),Math.sin(a)*Math.cos(b),Math.sin(b)];
    if(ring===1){[gp[1],gp[2]]=[gp[2],gp[1]];[gn[1],gn[2]]=[gn[2],gn[1]];}else if(ring===2){[gp[0],gp[2]]=[gp[2],gp[0]];[gn[0],gn[2]]=[gn[2],gn[0]];}
    surface(9,i,...gp,...gn);
    seeds.set([rand(),rand(),rand(),rand()],i*4);
  }
  current.set(forms[0]);origin.set(current);
  let w=0,h=0,scale=1,shape=0,sim=0,last=0,raf=0,morphStart=-5,nextMorph=8;
  let paused=reduced.matches,visible=true,yaw=.52,pitch=-.24,spin=0,vx=0,vy=0;
  let pointerX=0,pointerY=0,lightX=-.4,lightY=-.55,hover=0,present=false;
  let down=false,dragged=false,dragDistance=0,prevX=0,prevY=0,prevTime=0,touch=false;
  let effect=null,clickDir=[0,0,1];
  function describe(){label.textContent=`${String(shape+1).padStart(2,'0')} / 10 · ${names[shape]} ↗`;hint.textContent=`${quirks[shape]} · drag to rotate`;sculpture.setAttribute('aria-label',`${names[shape]} particles. ${quirks[shape]}. Drag to rotate; arrow keys change form.`);canvas.dataset.shape=names[shape];}
  function setShape(next,immediate=false){
    next=(next+names.length)%names.length;if(next===shape&&!immediate)return;
    origin.set(current);shape=next;morphStart=sim;nextMorph=sim+10;effect=null;describe();
    if(paused||immediate){current.set(forms[shape]);morphStart=sim-5;}
    if(paused)render(0);
  }
  function resize(){const rect=canvas.getBoundingClientRect(),dpr=Math.min(devicePixelRatio||1,2);w=rect.width;h=rect.height;scale=Math.min(w*.34,h*.35);canvas.width=Math.round(w*dpr);canvas.height=Math.round(h*dpr);ctx.setTransform(dpr,0,0,dpr,0,0);render(0);}
  function lightPointer(e){const r=canvas.getBoundingClientRect();pointerX=clamp((e.clientX-r.left-w/2)/scale,-1.6,1.6);pointerY=clamp((e.clientY-r.top-h/2)/scale,-1.6,1.6);present=true;}
  function energise(e){
    if(dragged){dragged=false;return;}if(paused)return;
    if(e.detail!==0)lightPointer(e);
    const px=e.detail===0?0:pointerX,py=e.detail===0?0:pointerY;
    const len=Math.hypot(px,py),pz=Math.sqrt(Math.max(.05,1-Math.min(.95,len*len)));
    const cy=Math.cos(yaw),sy=Math.sin(yaw),cx=Math.cos(pitch),sx=Math.sin(pitch),yy=py*cx+pz*sx,zz=-py*sx+pz*cx;
    clickDir=[px*cy-zz*sy,yy,px*sy+zz*cy];const norm=Math.hypot(...clickDir);clickDir=clickDir.map(v=>v/norm);
    const kind=['scatter','tumble','ripple','vortex','flex','twist','singularity','unzip','knot-pulse','gimbal'][shape];
    effect={kind,start:sim};canvas.dataset.effect=kind;
    spin=Math.min(4,spin+(kind==='tumble'?2.1:kind==='vortex'?1.25:.18));
    if(kind==='tumble')vy+=py>=0?.45:-.45;
    nextMorph=sim+10;
  }
  function render(dt){
    const ease=1-Math.exp(-dt*5);
    lightX+=((present?pointerX:-.4)-lightX)*ease;lightY+=((present?pointerY:-.55)-lightY)*ease;hover+=((present?1:0)-hover)*ease;
    if(!paused){if(!down){yaw+=dt*(.075+spin+vx);pitch+=vy*dt;vx*=Math.exp(-dt*1.7);vy*=Math.exp(-dt*1.7);}spin*=Math.exp(-dt*1.05);if(sim>=nextMorph&&!down&&!effect)setShape(shape+1);}
    const cy=Math.cos(yaw),sy=Math.sin(yaw),cx=Math.cos(pitch),sx=Math.sin(pitch),lx=lightX*.9,ly=lightY*.9,lz=.8,ln=Math.hypot(lx,ly,lz);
    const elapsed=effect?sim-effect.start:100,kind=effect?.kind;
    if(effect&&elapsed>3.8){effect=null;canvas.dataset.effect='idle';}
    const envelope=Math.sin(clamp(elapsed/3.1)*Math.PI)**2;
    const gravity=kind==='singularity'?envelope:0,unzip=kind==='unzip'?envelope:0,knotPulse=kind==='knot-pulse'?envelope:0,gimbal=kind==='gimbal'?envelope:0;
    const blast=kind==='scatter'?envelope:0,swirl=['vortex','twist'].includes(kind)?envelope:0,flex=kind==='flex'?Math.sin(elapsed*5)*Math.exp(-elapsed*.9):0;
    ctx.clearRect(0,0,w,h);for(const b of buckets)b.length=0;
    for(let i=0;i<count;i++){
      const o=i*6,s=i*4,j=i*5,t=smooth((sim-morphStart-seeds[s]*.3)/1.9);
      if(t<1&&!paused){for(let k=0;k<6;k++)current[o+k]=mix(origin[o+k],forms[shape][o+k],t);}else if(t>=1){for(let k=0;k<6;k++)current[o+k]=forms[shape][o+k];}
      let x=current[o],y=current[o+1],z=current[o+2],nx=current[o+3],ny=current[o+4],nz=current[o+5];
      let specialLight=0,specialAlpha=1;
      if(shape===6){
        const rad=Math.hypot(x,y),phase=Math.atan2(y,x),rate=.13/Math.pow(rad+.25,1.5);
        const theta=phase+t*(sim*rate+gravity*(1.1+(1-rad)*3.5));
        const radius=rad*(1-gravity*.69*t),rimLight=Math.exp(-Math.pow((rad-.43)/.16,2));
        x=mix(x,radius*Math.cos(theta),t);y=mix(y,radius*Math.sin(theta),t);z+=t*gravity*.34*Math.sin(theta*2);
        specialLight=t*(rimLight*.48+gravity*.3+(.5+.5*Math.sin(theta*3-sim*.65))*rimLight*.22);
        specialAlpha=1+t*.4;
      }
      if(shape===7&&unzip){
        const side=i%2?1:-1,angle=unzip*side*(.55+Math.abs(y)*.65),c=Math.cos(angle),sine=Math.sin(angle),old=x;
        x=(x*c-z*sine)*(1+unzip*.5);z=(old*sine+z*c)*(1+unzip*.5);
        if(i%5===0)specialAlpha=1-unzip*.75;
        specialLight=unzip*.2;
      }
      if(shape===8&&knotPulse){
        const progress=((i*.618033988749895)%1)*tau,travel=elapsed*3.8;
        const pulse=Math.exp(-Math.pow(Math.sin((progress-travel)/2)/.18,2))*knotPulse;
        x+=nx*pulse*.21;y+=ny*pulse*.21;z+=nz*pulse*.21;specialLight=pulse*.85;
      }
      if(shape===9){
        const ring=i%3,turn=t*(sim*(.13+ring*.07)+gimbal*(4+ring*1.8)),c=Math.cos(turn),ss=Math.sin(turn);
        let old,oldn;
        if(ring===0){old=y;oldn=ny;y=y*c-z*ss;z=old*ss+z*c;ny=ny*c-nz*ss;nz=oldn*ss+nz*c;}
        else if(ring===1){old=x;oldn=nx;x=x*c-z*ss;z=old*ss+z*c;nx=nx*c-nz*ss;nz=oldn*ss+nz*c;}
        else{old=x;oldn=nx;x=x*c-y*ss;y=old*ss+y*c;nx=nx*c-ny*ss;ny=oldn*ss+ny*c;}
        specialLight=.12+gimbal*.3;
      }
      const nl=Math.hypot(nx,ny,nz)||1;nx/=nl;ny/=nl;nz/=nl;
      const length=Math.hypot(x,y,z)||1,distance=effect?Math.acos(clamp((x*clickDir[0]+y*clickDir[1]+z*clickDir[2])/length,-1,1)):0;
      const wave=effect?Math.exp(-(((distance-elapsed*2.6)/.24)**2))*Math.exp(-elapsed*.72):0;
      const breathing=1+.009*Math.sin(sim*.78),scatter=blast*(.5+seeds[s+1]*.75),pulse=wave*(kind==='ripple'?.14:.025);
      x=x*breathing+nx*pulse+x*scatter;y=y*breathing+ny*pulse+y*scatter;z=z*breathing+nz*pulse+z*scatter;
      if(blast){x+=Math.sin(seeds[s+2]*tau)*blast*.17;y+=Math.cos(seeds[s+3]*tau)*blast*.17;}
      if(flex){const pinch=1-flex*.36*Math.exp(-y*y*4);x*=pinch;z*=pinch;y*=1+flex*.16;}
      if(swirl){const twist=swirl*(kind==='twist'?y*5+Math.sin(Math.atan2(z,x))*1.3:y*3+.6),c=Math.cos(twist),ss=Math.sin(twist),old=x,oldn=nx;x=x*c-z*ss;z=old*ss+z*c;nx=nx*c-nz*ss;nz=oldn*ss+nz*c;if(kind==='vortex')y+=Math.sin(Math.atan2(z,x)*3-elapsed*5)*swirl*.17;}
      const rx=x*cy+z*sy,rz=-x*sy+z*cy,ry=y*cx-rz*sx,depth=y*sx+rz*cx;
      const nrx=nx*cy+nz*sy,nrz=-nx*sy+nz*cy,nry=ny*cx-nrz*sx,nrz2=ny*sx+nrz*cx;
      const front=clamp((nrz2+.35)/1.35),lambert=Math.max(0,(nrx*lx+nry*ly+nrz2*lz)/ln),rim=clamp(1-Math.abs(nrz2))**2.4,sheen=lambert**14*hover;
      const scan=Math.exp(-(((y-Math.sin(sim*.36)*1.4)/.15)**2))*.1;
      let alpha=(.045+front*.21+rim*.3+lambert*.25+sheen*.4+wave*.7+scan)*(.73+seeds[s+3]*.27);
      if(shape===5)alpha=Math.max(alpha,.1+Math.abs(nrz2)*.24);
      alpha=clamp((alpha+specialLight)*specialAlpha*(1-blast*.23),.025,.98);
      const perspective=4.8/(4.8-depth),radius=(.42+seeds[s+2]*.4+rim*.18+sheen*.22+wave*.35)*perspective;
      rendered[j]=w/2+rx*scale*perspective/(1+blast*.55);rendered[j+1]=h/2+ry*scale*perspective/(1+blast*.55);rendered[j+2]=radius;rendered[j+3]=alpha;rendered[j+4]=sheen+wave*.5+specialLight*.6;
      buckets[Math.floor(clamp((depth+2.5)/5)*39)].push(i);
    }
    for(const bucket of buckets){for(const i of bucket){const j=i*5;ctx.globalAlpha=rendered[j+3];ctx.fillStyle=rendered[j+4]>.35?'#d1e0ff':'#91a0ec';ctx.beginPath();ctx.arc(rendered[j],rendered[j+1],rendered[j+2],0,tau);ctx.fill();}}ctx.globalAlpha=1;
  }
  function tick(now){raf=0;if(paused||document.hidden||!visible)return;const dt=Math.min(.04,(now-last)/1000||.016);last=now;sim+=dt;render(dt);raf=requestAnimationFrame(tick);}
  function start(){cancelAnimationFrame(raf);raf=0;if(!paused&&!document.hidden&&visible){last=performance.now();raf=requestAnimationFrame(tick);}}
  function motionLabel(){motion.textContent=paused?'Resume motion':'Pause motion';motion.setAttribute('aria-pressed',String(paused));motion.setAttribute('aria-label',paused?'Resume particle animation':'Pause particle animation');canvas.dataset.paused=String(paused);}
  sculpture.addEventListener('click',energise);label.addEventListener('click',()=>setShape(shape+1));
  sculpture.addEventListener('pointerdown',e=>{if(e.button!==0||paused)return;down=true;dragged=false;dragDistance=0;prevX=e.clientX;prevY=e.clientY;prevTime=e.timeStamp;touch=e.pointerType==='touch';vx=vy=0;lightPointer(e);if(!touch)sculpture.setPointerCapture(e.pointerId);});
  sculpture.addEventListener('pointermove',e=>{lightPointer(e);if(down){const dx=e.clientX-prevX,dy=e.clientY-prevY,dt=Math.max(.008,(e.timeStamp-prevTime)/1000);dragDistance+=Math.abs(dx)+Math.abs(dy);dragged=dragDistance>6;yaw+=dx*.007;pitch-=dy*.007;vx=clamp(dx*.007/dt,-3,3);vy=clamp(-dy*.007/dt,-3,3);prevX=e.clientX;prevY=e.clientY;prevTime=e.timeStamp;nextMorph=sim+8;}});
  const release=e=>{down=false;if(sculpture.hasPointerCapture(e.pointerId))sculpture.releasePointerCapture(e.pointerId);if(touch)present=false;};
  sculpture.addEventListener('pointerup',release);sculpture.addEventListener('pointercancel',e=>{release(e);dragged=false;vx=vy=0;});
  sculpture.addEventListener('lostpointercapture',()=>{down=false;});sculpture.addEventListener('pointerleave',()=>{present=false;if(touch)down=false;});
  sculpture.addEventListener('keydown',e=>{if(['ArrowRight','ArrowLeft'].includes(e.key)){e.preventDefault();setShape(shape+(e.key==='ArrowRight'?1:-1));}if(e.key==='Escape'){effect=null;spin=vx=vy=0;}});
  motion.addEventListener('click',()=>{paused=!paused;down=false;motionLabel();start();});
  reduced.addEventListener('change',()=>{paused=reduced.matches;effect=null;spin=vx=vy=0;motionLabel();start();});
  document.addEventListener('visibilitychange',start);
  new IntersectionObserver(entries=>{visible=entries[0].isIntersecting;start();},{rootMargin:'80px'}).observe(canvas);
  new ResizeObserver(resize).observe(canvas);describe();motionLabel();resize();start();
  const nav=[...document.querySelectorAll('nav a')],sectionForms={overview:0,experience:1,practice:3,writing:4,contact:5};
  const observer=new IntersectionObserver(entries=>{for(const entry of entries){if(!entry.isIntersecting)continue;nav.forEach(a=>{const active=a.hash===`#${entry.target.id}`;a.classList.toggle('active',active);if(active)a.setAttribute('aria-current','location');else a.removeAttribute('aria-current');});if(!mobile.matches&&!paused)setShape(sectionForms[entry.target.id]||0);}},{rootMargin:'-32% 0px -43% 0px',threshold:0});
  document.querySelectorAll('.chapter').forEach(section=>observer.observe(section));
})();



