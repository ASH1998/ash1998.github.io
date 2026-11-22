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
  const canvas=document.querySelector('#particles'),renderer=window.createParticleRenderer(canvas);
  const sculpture=document.querySelector('#sculpture'),label=document.querySelector('#shape-name'),motion=document.querySelector('#motion');
  const hint=document.querySelector('.interaction-hint');
  const reduced=matchMedia('(prefers-reduced-motion: reduce)'),mobile=matchMedia('(max-width:700px)');
  const count=mobile.matches?3456:7776;
  // The sphere is the page's form; the others appear only when the visitor asks for them via the label.
  // Eight forms, one for each year of experience.
  const names=['Sphere','Saturn','Atom','Wormhole','Lorenz','Network','Tesseract','Gyroid'];
  const order=names.map((_,i)=>i);
  const quirks=['Click to disperse','Click to pull the ring in','Click to blur the orbits into a cloud','Click to open the throat','Click for the butterfly effect','Click to run a forward and backward pass','Click to turn it inside out','Click to split the membrane'];
  const kinds=['scatter','accrete','cloud','warp','butterfly','pass','hyper','split'];
  const forms=names.map(()=>new Float32Array(count*6));
  const current=new Float32Array(count*6),origin=new Float32Array(count*6),visibleGeometry=new Float32Array(count*6);
  const seeds=new Float32Array(count*4),rendered=new Float32Array(count*5),buckets=Array.from({length:40},()=>[]);
  function surface(form,i,x,y,z,nx,ny,nz){const d=Math.hypot(nx,ny,nz)||1;forms[form].set([x,y,z,nx/d,ny/d,nz/d],i*6);}
  // Deterministic per-particle hash for the extra forms (keeps rand()'s sequence, and so the sphere, unchanged).
  function hash(i,k){const s=Math.sin(i*12.9898+k*78.233)*43758.5453;return s-Math.floor(s);}
  // Lorenz attractor, integrated once and scaled to fit the stage; particles flow along it.
  const lorenzLength=4096,lorenz=new Float32Array(lorenzLength*3);
  {let qx=.1,qy=0,qz=0;for(let k=0;k<lorenzLength+400;k++){for(let s=0;s<3;s++){const dx=10*(qy-qx),dy=qx*(28-qz)-qy,dz=qx*qy-8/3*qz;qx+=dx*.0025;qy+=dy*.0025;qz+=dz*.0025;}
    if(k>=400){const o=(k-400)*3;lorenz[o]=qx/21;lorenz[o+1]=(qz-24)/23;lorenz[o+2]=qy/23;}}}
  function lorenzPoint(i,time){const f=hash(i,7)*lorenzLength+time*30,k0=Math.floor(f)%lorenzLength,k1=(k0+1)%lorenzLength,fr=f-Math.floor(f),a3=k0*3,b3=k1*3;
    return [mix(lorenz[a3],lorenz[b3],fr)+(hash(i,8)-.5)*.035,mix(lorenz[a3+1],lorenz[b3+1],fr)+(hash(i,9)-.5)*.035,mix(lorenz[a3+2],lorenz[b3+2],fr)+(hash(i,10)-.5)*.035];}
  // A small feed-forward network: node clusters in five layers, joined by particle links.
  const layers=[3,5,6,5,3],netNodes=[],netLinks=[];
  layers.forEach((n,l)=>{const radius=n===3?.3:n===5?.48:.58;for(let j=0;j<n;j++){const ang=tau*j/n+l*.35;netNodes.push({l,p:[-1+l*.5,radius*Math.cos(ang),radius*Math.sin(ang)]});}});
  netNodes.forEach((a,ai)=>netNodes.forEach((b,bi)=>{if(b.l===a.l+1)netLinks.push([ai,bi]);}));
  const netPhase=new Float32Array(count);
  // An evenly spread unit direction per particle (the sphere's own lattice), reused for small clusters.
  function fibDir(i){const u=(i*.618033988749895)%1,v=(i+.5)/count,z=1-2*v,r=Math.sqrt(1-z*z),a=tau*u;return [r*Math.cos(a),z,r*Math.sin(a)];}
  // Saturn's moon: a small sphere on its own inclined orbit just outside the ring.
  function moonPoint(i,phase){const d=fibDir(i),c=[1.6*Math.cos(phase),1.6*Math.sin(phase)*.45,1.6*Math.sin(phase)*.89];return [c[0]+d[0]*.085,c[1]+d[1]*.085,c[2]+d[2]*.085];}
  // Atom: three tilted circular orbits (120 degrees apart) around a lumpy nucleus.
  const atomBasis=[0,1,2].map(k=>{const p=k*tau/3,t=1.15;return [Math.cos(p),0,-Math.sin(p),Math.cos(t)*Math.sin(p),-Math.sin(t),Math.cos(t)*Math.cos(p)];});
  function orbitPoint(k,th,R=.95){const e=atomBasis[k];return [R*(Math.cos(th)*e[0]+Math.sin(th)*e[3]),R*(Math.cos(th)*e[1]+Math.sin(th)*e[4]),R*(Math.cos(th)*e[2]+Math.sin(th)*e[5])];}
  // Tesseract: the 16 vertices and 32 edges of a 4D hypercube, rotated in 4D and projected to 3D.
  const tessVerts=[];for(let k=0;k<16;k++)tessVerts.push([k&1?1:-1,k&2?1:-1,k&4?1:-1,k&8?1:-1]);
  const tessEdges=[];for(let a=0;a<16;a++)for(let b=a+1;b<16;b++)if([0,1,2,3].filter(k=>tessVerts[a][k]!==tessVerts[b][k]).length===1)tessEdges.push([a,b]);
  function tessPoint(i,phase){
    const vertex=i%20<3,p4=vertex?tessVerts[Math.floor(hash(i,31)*16)]:(()=>{const [a,b]=tessEdges[i%tessEdges.length],f=hash(i,30);return tessVerts[a].map((e,k)=>mix(e,tessVerts[b][k],f));})();
    const ca=Math.cos(phase),sa=Math.sin(phase),cb=Math.cos(phase*.6),sb=Math.sin(phase*.6);
    const x=p4[0]*ca-p4[3]*sa,w1=p4[0]*sa+p4[3]*ca,z=p4[2]*cb-w1*sb,w=p4[2]*sb+w1*cb,s=1.55/(3-w);
    const d=vertex?fibDir(i):[(hash(i,32)-.5)*.3,(hash(i,33)-.5)*.3,(hash(i,34)-.5)*.3],j=vertex?.035:.03;
    return [x*s+d[0]*j,p4[1]*s+d[1]*j,z*s+d[2]*j];
  }
  // Gyroid: the triply periodic minimal surface sin x cos y + sin y cos z + sin z cos x = 0, clipped to a ball.
  function gyroidPoint(i){
    for(let attempt=0;attempt<24;attempt++){
      const k=40+attempt*3;let q=[hash(i,k)*2-1,hash(i,k+1)*2-1,hash(i,k+2)*2-1];if(Math.hypot(...q)>1)continue;
      q=q.map(e=>e*3.2);let g=[0,0,1];
      for(let n=0;n<8;n++){const [a,b,c]=q,f=Math.sin(a)*Math.cos(b)+Math.sin(b)*Math.cos(c)+Math.sin(c)*Math.cos(a);
        g=[Math.cos(a)*Math.cos(b)-Math.sin(c)*Math.sin(a),-Math.sin(a)*Math.sin(b)+Math.cos(b)*Math.cos(c),-Math.sin(b)*Math.sin(c)+Math.cos(c)*Math.cos(a)];
        const g2=g[0]*g[0]+g[1]*g[1]+g[2]*g[2]||1;q=q.map((e,m)=>e-f*g[m]/g2);}
      const p=q.map(e=>e/3.2);if(Math.hypot(...p)<=1)return [...p,...g];
    }
    const d=fibDir(i);return [...d,...d];
  }
  // Wormhole: two flared funnels joined by a narrow throat, drawn as a grid, with travellers spiralling through.
  function wormRadius(y,throat=.2){return throat+.95*(y/.9)**2;}
  function wormPoint(i,phase,throat=.2){
    const role=i%20;let y,ph,inset=1;
    if(role<6){y=-.9+Math.floor(hash(i,70)*16)*(1.8/15);ph=tau*hash(i,71);}            // 16 rings
    else if(role<11){ph=Math.floor(hash(i,72)*20)*tau/20;y=hash(i,73)*1.8-.9;}         // 20 meridians
    else if(role<13){y=(hash(i,74)-.5)*.05;ph=tau*hash(i,75);}                          // glowing throat ring
    else{const f=(hash(i,76)+phase)%1;y=.9-1.8*f;ph=tau*hash(i,77)+f*tau*2.2;inset=.9;}  // travellers
    const r=wormRadius(y,throat)*inset,dr=2*.95*y/.81,jit=(hash(i,78)-.5)*.008;
    return [(r+jit)*Math.cos(ph),y,(r+jit)*Math.sin(ph),Math.cos(ph),-dr,Math.sin(ph)];
  }
  for(let i=0;i<count;i++){
    const u=(i*.618033988749895)%1,v=(i+.5)/count,z=1-2*v,r=Math.sqrt(1-z*z),a=tau*u;
    const x=r*Math.cos(a),y=z,zz=r*Math.sin(a);
    surface(0,i,x,y,zz,x,y,zz);
    rand();rand();rand(); // the retired forms drew three values here; keep the sequence so the sphere's particles look the same
    // Saturn: a smaller planet, one moon, and a tilted ring with a Cassini-style gap.
    if(i%20<12)surface(1,i,x*.6,y*.6,zz*.6,x,y,zz);
    else if(i%20===12)surface(1,i,...moonPoint(i,0),x,y,zz);
    else{const ra=tau*hash(i,2);let rr=.86+.5*hash(i,3);if(rr>1.05&&rr<1.1)rr+=.06;const ry=(hash(i,4)-.5)*.012,c=Math.cos(.8),s=Math.sin(.8),c2=Math.cos(.32),s2=Math.sin(.32),rx=rr*Math.cos(ra),rz=rr*Math.sin(ra);
      // Tilt the ring towards the viewer (then roll it a little) so it reads as an ellipse, not a line.
      const ty=ry*c-rz*s,tz=ry*s+rz*c;surface(1,i,rx*c2-ty*s2,rx*s2+ty*c2,tz,-c*s2,c*c2,s);}
    // Atom: 20% a lumpy nucleus (eight small spheres), 15% three electrons, 65% their orbit paths.
    {const role=i%20;
      if(role<4){const o=[(i%8&1)?.08:-.08,(i%8&2)?.08:-.08,(i%8&4)?.08:-.08];surface(2,i,o[0]+x*.1,o[1]+y*.1,o[2]+zz*.1,x,y,zz);}
      else if(role<7){const c=orbitPoint(role-4,(role-4)*2.1);surface(2,i,c[0]+x*.05,c[1]+y*.05,c[2]+zz*.05,x,y,zz);}
      else{const p=orbitPoint(i%3,tau*hash(i,20)).map((e,m)=>e+(hash(i,21+m)-.5)*.012);surface(2,i,...p,...p);}}
    // Wormhole: grid, throat ring and travellers (the travellers move in render()).
    surface(3,i,...wormPoint(i,0));
    // Lorenz: particles placed along the attractor (they flow along it in render()).
    {const p=lorenzPoint(i,0),wing=p[0]>0?[.4,.13,.37]:[-.4,.13,-.37];surface(4,i,...p,p[0]-wing[0],p[1]-wing[1],p[2]-wing[2]);}
    // Network: 45% of particles form node clusters, the rest dotted links between layers.
    if(i%20<9){const node=netNodes[Math.floor(hash(i,11)*netNodes.length)];surface(5,i,node.p[0]+x*.075,node.p[1]+y*.075,node.p[2]+zz*.075,x,y,zz);netPhase[i]=node.l/4;}
    else{const [ai,bi]=netLinks[i%netLinks.length],A=netNodes[ai],B=netNodes[bi],f=hash(i,12),p=A.p.map((e,k)=>mix(e,B.p[k],f)+(hash(i,13+k)-.5)*.008);
      surface(5,i,...p,p[0]*.2,p[1],p[2]);netPhase[i]=(A.l+f)/4;}
    // Tesseract (animated in render()) and Gyroid (static surface; its click splits it into two sheets).
    {const p=tessPoint(i,0);surface(6,i,...p,...p);}
    surface(7,i,...gyroidPoint(i));
    seeds.set([rand(),rand(),rand(),rand()],i*4);
  }
  current.set(forms[0]);origin.set(current);visibleGeometry.set(current);
  // Slow interaction timelines only; rotation and idle motion use unscaled time.
  const interactionDurationScale=1.6;
  const morphDuration=2.5*interactionDurationScale,morphStagger=.2*interactionDurationScale;
  let w=0,h=0,scale=1,shape=0,sim=0,last=0,raf=0,morphStart=0,morphing=false;
  let paused=reduced.matches,visible=true,yaw=.52,pitch=-.24,spin=0,spinGoal=0,vx=0,vy=0,dragYaw=.52,dragPitch=-.24;
  let pointerX=0,pointerY=0,lightX=-.4,lightY=-.55,hover=0,present=false;
  let down=false,dragged=false,dragDistance=0,prevX=0,prevY=0,touch=false;
  let effect=null,clickDir=[0,0,1];
  let moonPhase=0,atomPhase=0,tessPhase=0,wormPhase=0;
  function describe(){const form=order[shape],total=String(names.length).padStart(2,'0'),next=names[order[(shape+1)%names.length]];label.textContent=`${String(shape+1).padStart(2,'0')} / ${total} · ${names[form]} ↗`;hint.textContent=`${quirks[form]} · drag to rotate`;sculpture.setAttribute('aria-label',`${names[form]} particles. ${quirks[form]}. Drag to rotate; arrow keys change the form.`);label.setAttribute('aria-label',`Show the next particle form: ${next}`);label.title=`Next: ${next}`;canvas.dataset.shape=names[form];}
  function setShape(next,immediate=false){
    next=(next+names.length)%names.length;if(next===shape&&!immediate)return;
    origin.set(visibleGeometry);shape=next;morphStart=sim;effect=null;canvas.dataset.effect='idle';morphing=!paused&&!immediate;describe();
    if(!morphing)current.set(forms[order[shape]]);
    if(paused)render(0);
  }
  function resize(){const rect=canvas.getBoundingClientRect(),dpr=Math.min(devicePixelRatio||1,2);w=rect.width;h=rect.height;scale=Math.min(w*.34,h*.35);canvas.width=Math.round(w*dpr);canvas.height=Math.round(h*dpr);renderer.resize(w,h,dpr);render(0);}
  function lightPointer(e){const r=canvas.getBoundingClientRect();pointerX=clamp((e.clientX-r.left-w/2)/scale,-1.6,1.6);pointerY=clamp((e.clientY-r.top-h/2)/scale,-1.6,1.6);present=true;}
  function energise(e){
    if(dragged){dragged=false;return;}if(paused)return;
    if(e.detail!==0)lightPointer(e);
    const px=e.detail===0?0:pointerX,py=e.detail===0?0:pointerY;
    const len=Math.hypot(px,py),pz=Math.sqrt(Math.max(.05,1-Math.min(.95,len*len)));
    const cy=Math.cos(yaw),sy=Math.sin(yaw),cx=Math.cos(pitch),sx=Math.sin(pitch),yy=py*cx+pz*sx,zz=-py*sx+pz*cx;
    clickDir=[px*cy-zz*sy,yy,px*sy+zz*cy];const norm=Math.hypot(...clickDir);clickDir=clickDir.map(v=>v/norm);
    const kind=kinds[order[shape]];
    effect={kind,start:sim};canvas.dataset.effect=kind;
    spinGoal=Math.min(4,spinGoal+(kind==='tumble'?2.1:kind==='vortex'?1.25:.18));
    if(kind==='tumble')vy+=py>=0?.45:-.45;
  }
  function extraShape(direction=1){setShape(shape+direction);}
  function render(dt){
    const ease=1-Math.exp(-dt*5);
    lightX+=((present?pointerX:-.4)-lightX)*ease;lightY+=((present?pointerY:-.55)-lightY)*ease;hover+=((present?1:0)-hover)*ease;
    if(!paused){
      const friction=Math.exp(-dt*1.7),spinDecay=Math.exp(-dt*6),goalDecay=Math.exp(-dt*.9);
      // Integrate damping over elapsed time so 60 Hz and 120 Hz feel the same.
      const spinTravel=spin*(1-spinDecay)/6+6*spinGoal/5.1*((1-goalDecay)/.9-(1-spinDecay)/6);
      if(down){const follow=1-Math.exp(-dt*14),dyaw=(dragYaw-yaw)*follow,dpitch=(dragPitch-pitch)*follow;yaw+=dyaw;pitch+=dpitch;if(dt>0){vx=clamp(dyaw/dt,-3,3);vy=clamp(dpitch/dt,-3,3);}}
      else{yaw+=dt*.06+spinTravel+vx*(1-friction)/1.7;pitch+=vy*(1-friction)/1.7;vx*=friction;vy*=friction;}
      spin=spin*spinDecay+6*spinGoal/5.1*(goalDecay-spinDecay);spinGoal*=goalDecay;
    }
    const form=order[shape],cy=Math.cos(yaw),sy=Math.sin(yaw),cx=Math.cos(pitch),sx=Math.sin(pitch),lx=lightX*.9,ly=lightY*.9,lz=.8,ln=Math.hypot(lx,ly,lz);
    const elapsed=effect?(sim-effect.start)/interactionDurationScale:100,kind=effect?.kind;
    if(effect&&elapsed>3.8){effect=null;canvas.dataset.effect='idle';}
    const envelope=Math.sin(clamp(elapsed/3.1)*Math.PI)**2;
    const accrete=kind==='accrete'?envelope:0,cloud=kind==='cloud'?envelope:0,hyper=kind==='hyper'?envelope:0,split=kind==='split'?envelope:0;
    const warp=kind==='warp'?envelope:0;
    const flap=kind==='butterfly'?Math.sin(elapsed*5)*envelope*1.15:0;
    const passFront=kind==='pass'?(elapsed<1.6?elapsed/1.6:Math.max(-1,1-(elapsed-1.6)/1.6)):-1;
    const blast=kind==='scatter'?envelope:0,swirl=['vortex','twist'].includes(kind)?envelope:0,flex=kind==='flex'?Math.sin(elapsed*5)*Math.exp(-elapsed*.9)*envelope:0;
    const shapeTime=sim-morphStart,breath=.009*Math.sin(sim*.78),scanY=Math.sin(sim*.36)*1.4;
    if(morphing&&shapeTime>=morphDuration+morphStagger){current.set(forms[form]);morphing=false;}
    if(!paused){moonPhase+=dt*.3;atomPhase+=dt*1.1;tessPhase+=dt*.22*(1+hyper*6);wormPhase+=dt*.08*(1+warp*6);}
    for(const b of buckets)b.length=0;
    for(let i=0;i<count;i++){
      const o=i*6,s=i*4,j=i*5,t=morphing?smooth((shapeTime-seeds[s]*morphStagger)/morphDuration):1;
      if(morphing){for(let k=0;k<6;k++)current[o+k]=mix(origin[o+k],forms[form][o+k],t);}
      let x=current[o],y=current[o+1],z=current[o+2],nx=current[o+3],ny=current[o+4],nz=current[o+5];
      let specialLight=0,specialAlpha=1;
      if(form===1){
        const role=i%20;
        if(role===12){const q=moonPoint(i,moonPhase);x=mix(x,q[0],t);y=mix(y,q[1],t);z=mix(z,q[2],t);specialLight=.1*t;}
        else if(role>12){
          // Accretion: the ring spirals in towards the planet and back out.
          specialLight=.2*t;
          if(accrete){const k=1-.3*accrete,th=accrete*2.6,c=Math.cos(th),sn=Math.sin(th),cx2=ny*z-nz*y,cy2=nz*x-nx*z,cz2=nx*y-ny*x;
            x=(x*c+cx2*sn)*k;y=(y*c+cy2*sn)*k;z=(z*c+cz2*sn)*k;specialLight+=accrete*.45;}
        }
        else if(accrete){const sw=1+.06*accrete;x*=sw;y*=sw;z*=sw;specialLight=accrete*accrete*.5;}
      }
      if(form===2){
        const role=i%20;
        if(role>=4&&role<7){
          // Electrons travel round their orbits.
          const k=role-4,c=orbitPoint(k,atomPhase+k*2.1),d=fibDir(i);
          x=mix(x,c[0]+d[0]*.05,t);y=mix(y,c[1]+d[1]*.05,t);z=mix(z,c[2]+d[2]*.05,t);specialLight=.5*t;specialAlpha=1-cloud*.6;
        }
        else if(role>=7){
          // Orbit paths glow behind each electron like a comet tail.
          let d=(atomPhase+(i%3)*2.1-tau*hash(i,20))%tau;if(d<0)d+=tau;specialLight=t*(Math.exp(-d*1.4)*.55+.04);
        }
        else specialLight=.15*t;
        if(cloud&&role>=4){
          // Quantum cloud: orbits blur into a fuzzy probability shell, then snap back.
          const dx=hash(i,40)-.5,dy=hash(i,41)-.5,dz=hash(i,42)-.5,dl=Math.hypot(dx,dy,dz)||1,rr=.95+(hash(i,43)-.5)*.4;
          x=mix(x,dx/dl*rr,cloud);y=mix(y,dy/dl*rr,cloud);z=mix(z,dz/dl*rr,cloud);specialLight+=cloud*.12;
        }
        if(cloud&&role<4)specialLight+=cloud*.3;
      }
      if(form===3){
        // Wormhole: travellers spiral down one funnel, through the throat and out the other; a click widens the throat.
        const role=i%20,q=wormPoint(i,wormPhase,.2+warp*.18);
        if(role>=13||warp){x=mix(x,q[0],t);y=mix(y,q[1],t);z=mix(z,q[2],t);}
        specialLight=t*(role>=11&&role<13?.45:role>=13?.3+warp*.4:.04);
      }
      if(form===4){
        // Lorenz: particles flow along the attractor; a click makes its two wings flap.
        const q=lorenzPoint(i,shapeTime);x=mix(x,q[0],t);y=mix(y,q[1],t);z=mix(z,q[2],t);specialLight=.12*t;
        if(flap){const ax=Math.abs(x);x=x*Math.cos(flap);z+=ax*Math.sin(flap);specialLight+=Math.abs(flap)*.2;}
      }
      if(form===5){
        // Network: a signal sweeps input to output; a click runs a fast forward pass, then a backward pass.
        const front=(shapeTime*.2)%1.3-.15,dd=netPhase[i]-front;specialLight=t*Math.exp(-((dd/.045)**2))*.85;
        if(passFront>=0){const dp=netPhase[i]-passFront;specialLight+=Math.exp(-((dp/.07)**2))*1.6;}
      }
      if(form===6){
        // Tesseract: continuous 4D rotation, projected to 3D; a click spins it through itself.
        const q=tessPoint(i,tessPhase);x=mix(x,q[0],t);y=mix(y,q[1],t);z=mix(z,q[2],t);specialLight=t*(i%20<3?.35:.1)+hyper*.25;
      }
      if(form===7&&split){
        // Gyroid: the membrane splits into two parallel sheets, then rejoins.
        const side=hash(i,60)>.5?1:-1,off=side*split*.09;x+=nx*off;y+=ny*off;z+=nz*off;specialLight=split*.2;
      }
      const nl=Math.hypot(nx,ny,nz)||1;nx/=nl;ny/=nl;nz/=nl;
      const length=Math.hypot(x,y,z)||1,distance=effect?Math.acos(clamp((x*clickDir[0]+y*clickDir[1]+z*clickDir[2])/length,-1,1)):0;
      const wave=kind==='scatter'?Math.exp(-(((distance-elapsed*2.6)/.24)**2))*Math.exp(-elapsed*.72):0;
      const breathing=1+t*breath,scatter=blast*(.5+seeds[s+1]*.75),pulse=wave*(kind==='ripple'?.14:.025);
      x=x*breathing+nx*pulse+x*scatter;y=y*breathing+ny*pulse+y*scatter;z=z*breathing+nz*pulse+z*scatter;
      if(blast){x+=Math.sin(seeds[s+2]*tau)*blast*.17;y+=Math.cos(seeds[s+3]*tau)*blast*.17;}
      if(flex){const pinch=1-flex*.36*Math.exp(-y*y*4);x*=pinch;z*=pinch;y*=1+flex*.16;}
      if(swirl){const twist=swirl*(kind==='twist'?y*5+Math.sin(Math.atan2(z,x))*1.3:y*3+.6),c=Math.cos(twist),ss=Math.sin(twist),old=x,oldn=nx;x=x*c-z*ss;z=old*ss+z*c;nx=nx*c-nz*ss;nz=oldn*ss+nz*c;if(kind==='vortex')y+=Math.sin(Math.atan2(z,x)*3-elapsed*5)*swirl*.17;}
      visibleGeometry[o]=x/(1+blast*.55);visibleGeometry[o+1]=y/(1+blast*.55);visibleGeometry[o+2]=z/(1+blast*.55);visibleGeometry[o+3]=nx;visibleGeometry[o+4]=ny;visibleGeometry[o+5]=nz;
      const rx=x*cy+z*sy,rz=-x*sy+z*cy,ry=y*cx-rz*sx,depth=y*sx+rz*cx;
      const nrx=nx*cy+nz*sy,nrz=-nx*sy+nz*cy,nry=ny*cx-nrz*sx,nrz2=ny*sx+nrz*cx;
      const front=clamp((nrz2+.35)/1.35),lambert=Math.max(0,(nrx*lx+nry*ly+nrz2*lz)/ln),rim=clamp(1-Math.abs(nrz2))**2.4,sheen=lambert**14*hover;
      const scan=Math.exp(-(((y-scanY)/.15)**2))*.1;
      let alpha=(.045+front*.21+rim*.3+lambert*.25+sheen*.4+wave*.7+scan)*(.73+seeds[s+3]*.27);
      if(form===2||form===4||form===5||form===6)alpha=Math.max(alpha,.14); // thin forms need a floor to stay visible
      alpha=clamp((alpha+specialLight)*specialAlpha*(1-blast*.23),.025,.98);
      const perspective=4.8/(4.8-depth),radius=(.42+seeds[s+2]*.4+rim*.18+sheen*.22+wave*.35)*perspective;
      rendered[j]=w/2+rx*scale*perspective/(1+blast*.55);rendered[j+1]=h/2+ry*scale*perspective/(1+blast*.55);rendered[j+2]=radius;rendered[j+3]=alpha;rendered[j+4]=sheen+wave*.5+specialLight*.6;
      buckets[Math.floor(clamp((depth+2.5)/5)*39)].push(i);
    }
    renderer.draw(rendered,buckets,count,w,h);
  }
  function tick(now){raf=0;if(paused||document.hidden||!visible)return;const dt=Math.min(.04,(now-last)/1000||.016);last=now;sim+=dt;render(dt);raf=requestAnimationFrame(tick);}
  function start(){cancelAnimationFrame(raf);raf=0;if(!paused&&!document.hidden&&visible){last=performance.now();raf=requestAnimationFrame(tick);}}
  function motionLabel(){motion.textContent=paused?'Resume motion':'Pause motion';motion.setAttribute('aria-pressed',String(paused));motion.setAttribute('aria-label',paused?'Resume particle animation':'Pause particle animation');canvas.dataset.paused=String(paused);}
  sculpture.addEventListener('click',energise);label.addEventListener('click',()=>extraShape());
  sculpture.addEventListener('pointerdown',e=>{if(e.button!==0||paused)return;down=true;dragYaw=yaw;dragPitch=pitch;dragged=false;dragDistance=0;prevX=e.clientX;prevY=e.clientY;touch=e.pointerType==='touch';vx=vy=0;lightPointer(e);if(!touch)sculpture.setPointerCapture(e.pointerId);});
  sculpture.addEventListener('pointermove',e=>{lightPointer(e);if(down){const dx=e.clientX-prevX,dy=e.clientY-prevY;dragDistance+=Math.abs(dx)+Math.abs(dy);dragged=dragDistance>6;dragYaw+=dx*.006;dragPitch-=dy*.006;prevX=e.clientX;prevY=e.clientY;}});
  const release=e=>{down=false;if(sculpture.hasPointerCapture(e.pointerId))sculpture.releasePointerCapture(e.pointerId);if(touch)present=false;};
  sculpture.addEventListener('pointerup',release);sculpture.addEventListener('pointercancel',e=>{release(e);dragged=false;vx=vy=0;});
  sculpture.addEventListener('lostpointercapture',()=>{down=false;});sculpture.addEventListener('pointerleave',()=>{present=false;if(touch)down=false;});
  sculpture.addEventListener('keydown',e=>{if(['ArrowRight','ArrowLeft'].includes(e.key)){e.preventDefault();extraShape(e.key==='ArrowRight'?1:-1);}if(e.key==='Escape'){effect=null;canvas.dataset.effect='idle';spin=spinGoal=vx=vy=0;}});
  motion.addEventListener('click',()=>{paused=!paused;down=false;motionLabel();start();});
  reduced.addEventListener('change',()=>{paused=reduced.matches;effect=null;canvas.dataset.effect='idle';spin=spinGoal=vx=vy=0;motionLabel();start();});
  document.addEventListener('visibilitychange',start);
  new IntersectionObserver(entries=>{visible=entries[0].isIntersecting;start();},{rootMargin:'80px'}).observe(canvas);
  new ResizeObserver(resize).observe(canvas);describe();motionLabel();resize();start();
  // Scrolling only updates the navigation highlight. The sculpture stays a sphere for the whole page;
  // other forms appear only when the visitor clicks the form label (or uses the arrow keys).
  const nav=[...document.querySelectorAll('nav a')];
  const observer=new IntersectionObserver(entries=>{for(const entry of entries){if(!entry.isIntersecting)continue;nav.forEach(a=>{const active=a.hash===`#${entry.target.id}`;a.classList.toggle('active',active);if(active)a.setAttribute('aria-current','location');else a.removeAttribute('aria-current');});}},{rootMargin:'-32% 0px -43% 0px',threshold:0});
  document.querySelectorAll('.chapter').forEach(section=>observer.observe(section));
})();
