const fs=require('fs'),assert=require('assert');require('./static/model-core.js');const C=globalThis.LiftingCore,fixture=JSON.parse(fs.readFileSync('browser-port-fixture.json','utf8'),(k,v)=>v===null?NaN:v);
for(const c of fixture.counter){let counter=new C.LiftCounter(c.fps);assert.deepStrictEqual(c.values.map((v,i)=>counter.update(v===null?NaN:v,i)),c.expected,c.name);}
let g=fixture.geometry,m=C.measure(g.C,g.xy,g.calib);assert(Math.abs(m.H-g.expected.H)<1e-8);assert(Math.abs(m.V-g.expected.V)<1e-8);assert.strictEqual(m.limit,g.expected.limit);assert(Math.abs(C.twist(g.world)-g.expected.twist)<1e-8);
console.log(`PASS: ${fixture.counter.length} clip count signals match Python frame-for-frame; H/V/twist/table match on real 3D coordinates`);
