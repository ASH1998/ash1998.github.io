(() => {
  'use strict';
  window.createParticleRenderer = canvas => {
    const gl=canvas.getContext('webgl',{alpha:true,antialias:false,depth:false,stencil:false,powerPreference:'low-power'});
    if(!gl){
      const ctx=canvas.getContext('2d');canvas.dataset.renderer='canvas';
      return {resize(w,h,dpr){ctx.setTransform(dpr,0,0,dpr,0,0);},draw(points,buckets,count,w,h){ctx.clearRect(0,0,w,h);for(const bucket of buckets)for(const i of bucket){const j=i*5;ctx.globalAlpha=points[j+3];ctx.fillStyle=points[j+4]>.35?'#d1e0ff':'#91a0ec';ctx.beginPath();ctx.arc(points[j],points[j+1],points[j+2],0,Math.PI*2);ctx.fill();}ctx.globalAlpha=1;}};
    }
    canvas.dataset.renderer='webgl';
    let program,buffer,resolution,pixelRatio,capacity=0,packed=new Float32Array(0),lost=false,dpr=1;
    function shader(type,source){const s=gl.createShader(type);gl.shaderSource(s,source);gl.compileShader(s);if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))throw new Error(gl.getShaderInfoLog(s));return s;}
    function initialise(){
      const vs=shader(gl.VERTEX_SHADER,`attribute vec2 point;attribute float radius;attribute float opacity;attribute float glow;uniform vec2 resolution;uniform float pixelRatio;varying float alpha;varying float light;void main(){vec2 p=point/resolution*2.0-1.0;gl_Position=vec4(p.x,-p.y,0.,1.);gl_PointSize=max(1.,radius*2.*pixelRatio+1.);alpha=opacity;light=clamp(glow,0.,1.);}`);
      const fs=shader(gl.FRAGMENT_SHADER,`precision mediump float;varying float alpha;varying float light;void main(){float d=length(gl_PointCoord-vec2(.5))*2.;float edge=1.-smoothstep(.64,1.,d);if(d>1.)discard;vec3 color=mix(vec3(.569,.627,.925),vec3(.82,.878,1.),smoothstep(.15,.8,light));gl_FragColor=vec4(color,alpha*edge);}`);
      program=gl.createProgram();gl.attachShader(program,vs);gl.attachShader(program,fs);gl.linkProgram(program);if(!gl.getProgramParameter(program,gl.LINK_STATUS))throw new Error(gl.getProgramInfoLog(program));gl.deleteShader(vs);gl.deleteShader(fs);gl.useProgram(program);
      buffer=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,buffer);capacity=0;
      for(const [name,size,offset] of [['point',2,0],['radius',1,8],['opacity',1,12],['glow',1,16]]){const loc=gl.getAttribLocation(program,name);gl.enableVertexAttribArray(loc);gl.vertexAttribPointer(loc,size,gl.FLOAT,false,20,offset);}
      resolution=gl.getUniformLocation(program,'resolution');pixelRatio=gl.getUniformLocation(program,'pixelRatio');gl.enable(gl.BLEND);gl.blendFuncSeparate(gl.SRC_ALPHA,gl.ONE_MINUS_SRC_ALPHA,gl.ONE,gl.ONE_MINUS_SRC_ALPHA);gl.clearColor(0,0,0,0);lost=false;
    }
    initialise();
    canvas.addEventListener('webglcontextlost',e=>{e.preventDefault();lost=true;});
    canvas.addEventListener('webglcontextrestored',initialise);
    return {resize(w,h,ratio){dpr=ratio;},draw(points,buckets,count,w,h){
      if(lost)return;
      if(packed.length!==count*5)packed=new Float32Array(count*5);
      let offset=0;for(const bucket of buckets)for(const i of bucket){const j=i*5;for(let k=0;k<5;k++)packed[offset++]=points[j+k];}
      gl.viewport(0,0,canvas.width,canvas.height);gl.clear(gl.COLOR_BUFFER_BIT);gl.useProgram(program);gl.uniform2f(resolution,w,h);gl.uniform1f(pixelRatio,dpr);gl.bindBuffer(gl.ARRAY_BUFFER,buffer);
      if(capacity!==packed.byteLength){gl.bufferData(gl.ARRAY_BUFFER,packed.byteLength,gl.DYNAMIC_DRAW);capacity=packed.byteLength;}
      gl.bufferSubData(gl.ARRAY_BUFFER,0,packed);gl.drawArrays(gl.POINTS,0,count);
    }};
  };
})();
