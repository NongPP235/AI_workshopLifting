importScripts('model-core.js');
const send=self.postMessage.bind(self);
let detector,cvReady,smoother,calib,pose,samples,counter,index,w,h,fps,height;
async function init(){
 send({type:'phase',message:'กำลังเริ่ม OpenCV WebAssembly'});
 if(!cvReady){importScripts('runtime/opencv.js');const module=globalThis.cv;if(!module.Mat)await new Promise((resolve,reject)=>{let timer=setTimeout(()=>reject(Error('โหลด OpenCV Web เกินเวลา')),60000),old=module.onRuntimeInitialized;module.onRuntimeInitialized=()=>{old?.();clearTimeout(timer);resolve();};});cvReady=module;if(typeof cvReady.solvePnP!=='function'||typeof cvReady.Rodrigues!=='function')throw Error('OpenCV Web ไม่รองรับ solvePnP/Rodrigues');}

}
self.onmessage=async({data})=>{try{if(data.type==='init'){await init();({w,h,fps,height}=data);smoother=new LiftingCore.Smoother(w,h);samples=[];calib=pose=null;index=0;counter=new LiftingCore.LiftCounter(fps);send({type:'ready'});return;}
 if(data.type==='frame'){const {xy,world,confidence}=data;let s=smoother.update(xy,world,confidence),m=null;
 let measurement=null,twist=NaN;
 if(!calib){samples.push(s);if(samples.length>=Math.round(1.5*fps)){calib=LiftingCore.calibrate(cvReady,samples,height,w,h);samples=[];}}
 else{let P=s.world.map(p=>LiftingCore.mul(p,100*calib.worldScale));pose=LiftingCore.solve(cvReady,P,s.xy,s.observed,calib.K,pose);if(pose)measurement=LiftingCore.measure(LiftingCore.camera3d(s.xy,P,pose,calib.K),s.xy,calib);twist=Math.abs(((LiftingCore.twist(s.world)-calib.twistBaseline+180)%360+360)%360-180);}
 if(measurement)measurement.table=LiftingCore.tableCells(measurement,calib);
 let lifts=counter.update(measurement?.V??NaN,index);send({type:'frame',index,time:(index+1)/fps,xy:s.xy,observed:s.observed,measurement:measurement?{...measurement,body:undefined}:null,twist,lifts,calibrating:!measurement&&!calib});index++;}
 }catch(error){send({type:'error',message:error.message||String(error)});}};
