
var ortWasmThreaded = (() => {
  var _scriptName = import.meta.url;
  
  return (
async function(moduleArg = {}) {
  var moduleRtn;

function aa(){f.buffer!=l.buffer&&m();return l}function n(){f.buffer!=l.buffer&&m();return ba}function r(){f.buffer!=l.buffer&&m();return ca}function u(){f.buffer!=l.buffer&&m();return da}function ea(){f.buffer!=l.buffer&&m();return fa}var w=Object.assign({},moduleArg),ha,x,ia=new Promise((a,b)=>{ha=a;x=b}),ja="object"==typeof window,y="function"==typeof importScripts,B="object"==typeof process&&"object"==typeof process.versions&&"string"==typeof process.versions.node,C=y&&"em-pthread"==self.name;
if(B){const {createRequire:a}=await import("module");var require=a(import.meta.url),D=require("worker_threads");global.Worker=D.Worker;C=(y=!D.ib)&&"em-pthread"==D.workerData}"use strict";w.mountExternalData=(a,b)=>{(w.Ra||(w.Ra=new Map)).set(a,b)};w.unmountExternalData=()=>{delete w.Ra};
var SharedArrayBuffer=globalThis.SharedArrayBuffer??(new WebAssembly.Memory({initial:0,maximum:0,shared:!0})).buffer.constructor,ka=Object.assign({},w),la="./this.program",E=(a,b)=>{throw b;},F="",ma,G,H;
if(B){var fs=require("fs"),na=require("path");F=require("url").fileURLToPath(new URL("./",import.meta.url));ma=(a,b)=>{a=I(a)?new URL(a):na.normalize(a);return fs.readFileSync(a,b?void 0:"utf8")};H=a=>{a=ma(a,!0);a.buffer||(a=new Uint8Array(a));return a};G=(a,b,c,d=!0)=>{a=I(a)?new URL(a):na.normalize(a);fs.readFile(a,d?void 0:"utf8",(g,h)=>{g?c(g):b(d?h.buffer:h)})};!w.thisProgram&&1<process.argv.length&&(la=process.argv[1].replace(/\\/g,"/"));process.argv.slice(2);E=(a,b)=>{process.exitCode=
a;throw b;}}else if(ja||y)y?F=self.location.href:"undefined"!=typeof document&&document.currentScript&&(F=document.currentScript.src),_scriptName&&(F=_scriptName),F.startsWith("blob:")?F="":F=F.substr(0,F.replace(/[?#].*/,"").lastIndexOf("/")+1),B||(ma=a=>{var b=new XMLHttpRequest;b.open("GET",a,!1);b.send(null);return b.responseText},y&&(H=a=>{var b=new XMLHttpRequest;b.open("GET",a,!1);b.responseType="arraybuffer";b.send(null);return new Uint8Array(b.response)}),G=(a,b,c)=>{var d=new XMLHttpRequest;
d.open("GET",a,!0);d.responseType="arraybuffer";d.onload=()=>{200==d.status||0==d.status&&d.response?b(d.response):c()};d.onerror=c;d.send(null)});B&&"undefined"==typeof performance&&(global.performance=require("perf_hooks").performance);var oa=console.log.bind(console),pa=console.error.bind(console);B&&(oa=(...a)=>fs.writeSync(1,a.join(" ")+"\n"),pa=(...a)=>fs.writeSync(2,a.join(" ")+"\n"));var qa=oa,K=pa;Object.assign(w,ka);ka=null;
if(C){var ra;if(B){var sa=D.parentPort;sa.on("message",b=>onmessage({data:b}));Object.assign(globalThis,{self:global,importScripts:()=>{},postMessage:b=>sa.postMessage(b),performance:global.performance||{now:Date.now}})}var ta=!1;K=function(...b){b=b.join(" ");B?fs.writeSync(2,b+"\n"):console.error(b)};self.alert=function(...b){postMessage({Wa:"alert",text:b.join(" "),jb:L()})};w.instantiateWasm=(b,c)=>new Promise(d=>{ra=g=>{g=new WebAssembly.Instance(g,ua());c(g);d()}});self.onunhandledrejection=
b=>{throw b.reason||b;};function a(b){try{var c=b.data,d=c.cmd;if("load"===d){let g=[];self.onmessage=h=>g.push(h);self.startWorker=()=>{postMessage({cmd:"loaded"});for(let h of g)a(h);self.onmessage=a};for(const h of c.handlers)if(!w[h]||w[h].proxy)w[h]=(...k)=>{postMessage({Wa:"callHandler",hb:h,args:k})},"print"==h&&(qa=w[h]),"printErr"==h&&(K=w[h]);f=c.wasmMemory;m();ra(c.wasmModule)}else if("run"===d){va(c.pthread_ptr,0,0,1,0,0);wa(c.pthread_ptr);xa();ya();ta||=!0;try{za(c.start_routine,c.arg)}catch(g){if("unwind"!=
g)throw g;}}else"cancel"===d?L()&&Aa(-1):"setimmediate"!==c.target&&("checkMailbox"===d?ta&&Ba():d&&(K(`worker: received unknown command ${d}`),K(c)))}catch(g){throw Ca(),g;}}self.onmessage=a}var f,Da,Ea=!1,M,l,ba,ca,da,N,fa;
function m(){var a=f.buffer;w.HEAP8=l=new Int8Array(a);w.HEAP16=new Int16Array(a);w.HEAPU8=ba=new Uint8Array(a);w.HEAPU16=new Uint16Array(a);w.HEAP32=ca=new Int32Array(a);w.HEAPU32=da=new Uint32Array(a);w.HEAPF32=new Float32Array(a);w.HEAPF64=fa=new Float64Array(a);w.HEAP64=N=new BigInt64Array(a);w.HEAPU64=new BigUint64Array(a)}
if(!C){if(w.wasmMemory)f=w.wasmMemory;else if(f=new WebAssembly.Memory({initial:256,maximum:65536,shared:!0}),!(f.buffer instanceof SharedArrayBuffer))throw K("requested a shared WebAssembly.Memory but the returned buffer is not a SharedArrayBuffer, indicating that while the browser has SharedArrayBuffer it does not have WebAssembly threads support - you may need to set a flag"),B&&K("(on node you may need: --experimental-wasm-threads --experimental-wasm-bulk-memory and/or recent version)"),Error("bad memory");
m()}var Fa=[],Ga=[],Ha=[],O=0,Ia=null,P=null;function Ja(){O--;if(0==O&&(null!==Ia&&(clearInterval(Ia),Ia=null),P)){var a=P;P=null;a()}}function Ka(a){a="Aborted("+a+")";K(a);Ea=!0;M=1;a=new WebAssembly.RuntimeError(a+". Build with -sASSERTIONS for more info.");x(a);throw a;}var La=a=>a.startsWith("data:application/octet-stream;base64,"),I=a=>a.startsWith("file://"),Ma;function Na(a){if(H)return H(a);throw"both async and sync fetching of the wasm failed";}
function Oa(a){if(ja||y){if("function"==typeof fetch&&!I(a))return fetch(a,{credentials:"same-origin"}).then(b=>{if(!b.ok)throw`failed to load wasm binary file at '${a}'`;return b.arrayBuffer()}).catch(()=>Na(a));if(G)return new Promise((b,c)=>{G(a,d=>b(new Uint8Array(d)),c)})}return Promise.resolve().then(()=>Na(a))}function Pa(a,b,c){return Oa(a).then(d=>WebAssembly.instantiate(d,b)).then(c,d=>{K(`failed to asynchronously prepare wasm: ${d}`);Ka(d)})}
function Qa(a,b){var c=Ma;return"function"!=typeof WebAssembly.instantiateStreaming||La(c)||I(c)||B||"function"!=typeof fetch?Pa(c,a,b):fetch(c,{credentials:"same-origin"}).then(d=>WebAssembly.instantiateStreaming(d,a).then(b,function(g){K(`wasm streaming compile failed: ${g}`);K("falling back to ArrayBuffer instantiation");return Pa(c,a,b)}))}
function ua(){Ra={b:Sa,C:Ta,g:Ua,T:Va,x:Wa,A:Xa,U:Ya,R:Za,K:$a,Q:ab,m:bb,y:cb,v:db,S:eb,w:fb,o:gb,X:hb,N:ib,t:jb,D:kb,i:lb,M:wa,W:mb,G:nb,H:ob,J:pb,E:qb,F:rb,s:sb,n:tb,j:ub,h:vb,p:wb,V:xb,u:yb,d:zb,e:Ab,r:Bb,z:Cb,q:Db,O:Eb,P:Fb,B:Gb,f:Hb,l:Ib,L:Jb,k:Kb,a:f,I:Lb,c:Mb};return{a:Ra}}
var Nb={817188:(a,b,c,d)=>{if("undefined"==typeof w||!w.Ra)return 1;a=Q(a>>>0);a.startsWith("./")&&(a=a.substring(2));a=w.Ra.get(a);if(!a)return 2;b>>>=0;c>>>=0;d>>>=0;if(b+c>a.byteLength)return 3;try{return n().set(a.subarray(b,b+c),d>>>0),0}catch{return 4}},817689:()=>"undefined"!==typeof wasmOffsetConverter};function Ob(a){this.name="ExitStatus";this.message=`Program terminated with exit(${a})`;this.status=a}
var Pb=a=>{a.terminate();a.onmessage=()=>{}},Sb=a=>{0==R.length&&(Qb(),Rb(R[0]));var b=R.pop();if(!b)return 6;S.push(b);T[a.Oa]=b;b.Oa=a.Oa;var c={cmd:"run",start_routine:a.$a,arg:a.Ya,pthread_ptr:a.Oa};B&&b.unref();b.postMessage(c,a.fb);return 0},U=0,V=(a,b,...c)=>{for(var d=2*c.length,g=Tb(),h=Ub(8*d),k=h>>>3,q=0;q<c.length;q++){var z=c[q];"bigint"==typeof z?(N[k+2*q]=1n,N[k+2*q+1]=z):(N[k+2*q]=0n,ea()[k+2*q+1>>>0]=z)}a=Vb(a,0,d,h,b);Zb(g);return a};
function $b(a){if(C)return V(0,1,a);M=a;if(!(0<U)){for(var b of S)Pb(b);for(b of R)Pb(b);R=[];S=[];T=[];w.onExit?.(a);Ea=!0}E(a,new Ob(a))}function ac(a){if(C)return V(1,0,a);Gb(a)}var Gb=a=>{M=a;if(C)throw ac(a),"unwind";$b(a)},R=[],S=[],bc=[],T={};function cc(){for(var a=w.numThreads-1;a--;)Qb();Fa.unshift(()=>{O++;dc(()=>Ja())})}var fc=a=>{var b=a.Oa;delete T[b];R.push(a);S.splice(S.indexOf(a),1);a.Oa=0;ec(b)};function ya(){bc.forEach(a=>a())}
var Rb=a=>new Promise(b=>{a.onmessage=h=>{h=h.data;var k=h.cmd;if(h.targetThread&&h.targetThread!=L()){var q=T[h.targetThread];q?q.postMessage(h,h.transferList):K(`Internal error! Worker sent a message "${k}" to target pthread ${h.targetThread}, but that thread no longer exists!`)}else if("checkMailbox"===k)Ba();else if("spawnThread"===k)Sb(h);else if("cleanupThread"===k)fc(T[h.thread]);else if("killThread"===k)h=h.thread,k=T[h],delete T[h],Pb(k),ec(h),S.splice(S.indexOf(k),1),k.Oa=0;else if("cancelThread"===
k)T[h.thread].postMessage({cmd:"cancel"});else if("loaded"===k)a.loaded=!0,B&&!a.Oa&&a.unref(),b(a);else if("alert"===k)alert(`Thread ${h.threadId}: ${h.text}`);else if("setimmediate"===h.target)a.postMessage(h);else if("callHandler"===k)w[h.handler](...h.args);else k&&K(`worker sent an unknown command ${k}`)};a.onerror=h=>{K(`${"worker sent an error!"} ${h.filename}:${h.lineno}: ${h.message}`);throw h;};B&&(a.on("message",h=>a.onmessage({data:h})),a.on("error",h=>a.onerror(h)));var c=[],d=["onExit"],
g;for(g of d)w.hasOwnProperty(g)&&c.push(g);a.postMessage({cmd:"load",handlers:c,wasmMemory:f,wasmModule:Da})});function dc(a){C?a():Promise.all(R.map(Rb)).then(a)}function Qb(){var a=new Worker(new URL(import.meta.url),{type:"module",workerData:"em-pthread",name:"em-pthread"});R.push(a)}
var gc=a=>{for(;0<a.length;)a.shift()(w)},xa=()=>{var a=L(),b=u()[a+52>>>2>>>0];a=u()[a+56>>>2>>>0];hc(b,b-a);Zb(b)},ic=[],jc,za=(a,b)=>{U=0;var c=ic[a];c||(a>=ic.length&&(ic.length=a+1),ic[a]=c=jc.get(a));a=c(b);0<U?M=a:Aa(a)};class kc{constructor(a){this.Ua=a-24}}var lc=0,mc=0;function Sa(a,b,c){a>>>=0;var d=new kc(a);b>>>=0;c>>>=0;u()[d.Ua+16>>>2>>>0]=0;u()[d.Ua+4>>>2>>>0]=b;u()[d.Ua+8>>>2>>>0]=c;lc=a;mc++;throw lc;}function nc(a,b,c,d){return C?V(2,1,a,b,c,d):Ta(a,b,c,d)}
function Ta(a,b,c,d){a>>>=0;b>>>=0;c>>>=0;d>>>=0;if("undefined"==typeof SharedArrayBuffer)return K("Current environment does not support SharedArrayBuffer, pthreads are not available!"),6;var g=[];if(C&&0===g.length)return nc(a,b,c,d);a={$a:c,Oa:a,Ya:d,fb:g};return C?(a.Wa="spawnThread",postMessage(a,g),0):Sb(a)}
var oc="undefined"!=typeof TextDecoder?new TextDecoder("utf8"):void 0,pc=(a,b,c)=>{b>>>=0;var d=b+c;for(c=b;a[c]&&!(c>=d);)++c;if(16<c-b&&a.buffer&&oc)return oc.decode(a.buffer instanceof SharedArrayBuffer?a.slice(b,c):a.subarray(b,c));for(d="";b<c;){var g=a[b++];if(g&128){var h=a[b++]&63;if(192==(g&224))d+=String.fromCharCode((g&31)<<6|h);else{var k=a[b++]&63;g=224==(g&240)?(g&15)<<12|h<<6|k:(g&7)<<18|h<<12|k<<6|a[b++]&63;65536>g?d+=String.fromCharCode(g):(g-=65536,d+=String.fromCharCode(55296|g>>
10,56320|g&1023))}}else d+=String.fromCharCode(g)}return d},Q=(a,b)=>(a>>>=0)?pc(n(),a,b):"";function Ua(a,b,c){return C?V(3,1,a,b,c):0}function Va(a,b){if(C)return V(4,1,a,b)}
var qc=a=>{for(var b=0,c=0;c<a.length;++c){var d=a.charCodeAt(c);127>=d?b++:2047>=d?b+=2:55296<=d&&57343>=d?(b+=4,++c):b+=3}return b},rc=(a,b,c,d)=>{c>>>=0;if(!(0<d))return 0;var g=c;d=c+d-1;for(var h=0;h<a.length;++h){var k=a.charCodeAt(h);if(55296<=k&&57343>=k){var q=a.charCodeAt(++h);k=65536+((k&1023)<<10)|q&1023}if(127>=k){if(c>=d)break;b[c++>>>0]=k}else{if(2047>=k){if(c+1>=d)break;b[c++>>>0]=192|k>>6}else{if(65535>=k){if(c+2>=d)break;b[c++>>>0]=224|k>>12}else{if(c+3>=d)break;b[c++>>>0]=240|k>>
18;b[c++>>>0]=128|k>>12&63}b[c++>>>0]=128|k>>6&63}b[c++>>>0]=128|k&63}}b[c>>>0]=0;return c-g},W=(a,b,c)=>rc(a,n(),b,c);function Wa(a,b){if(C)return V(5,1,a,b)}function Xa(a,b,c){if(C)return V(6,1,a,b,c)}function Ya(a,b,c){return C?V(7,1,a,b,c):0}function Za(a,b){if(C)return V(8,1,a,b)}function $a(a,b,c){if(C)return V(9,1,a,b,c)}function ab(a,b,c,d){if(C)return V(10,1,a,b,c,d)}function bb(a,b,c,d){if(C)return V(11,1,a,b,c,d)}function cb(a,b,c,d){if(C)return V(12,1,a,b,c,d)}
function db(a){if(C)return V(13,1,a)}function eb(a,b){if(C)return V(14,1,a,b)}function fb(a,b,c){if(C)return V(15,1,a,b,c)}var gb=()=>{Ka("")},hb=()=>1;function ib(a){va(a>>>0,!y,1,!ja,131072,!1);ya()}function wa(a){a>>>=0;"function"===typeof Atomics.gb&&(Atomics.gb(r(),a>>>2,a).value.then(Ba),a+=128,Atomics.store(r(),a>>>2,1))}
var Ba=()=>{var a=L();if(a&&(wa(a),a=sc,!Ea))try{if(a(),!(0<U))try{C?Aa(M):Gb(M)}catch(b){b instanceof Ob||"unwind"==b||E(1,b)}}catch(b){b instanceof Ob||"unwind"==b||E(1,b)}};function jb(a,b){a>>>=0;a==b>>>0?setTimeout(Ba):C?postMessage({targetThread:a,cmd:"checkMailbox"}):(a=T[a])&&a.postMessage({cmd:"checkMailbox"})}var tc=[];function kb(a,b,c,d,g){b>>>=0;d/=2;tc.length=d;c=g>>>0>>>3;for(g=0;g<d;g++)tc[g]=N[c+2*g]?N[c+2*g+1]:ea()[c+2*g+1>>>0];return(b?Nb[b]:uc[a])(...tc)}
function lb(a){a>>>=0;C?postMessage({cmd:"cleanupThread",thread:a}):fc(T[a])}function mb(a){B&&T[a>>>0].ref()}
function nb(a,b){a=-9007199254740992>a||9007199254740992<a?NaN:Number(a);b>>>=0;a=new Date(1E3*a);r()[b>>>2>>>0]=a.getUTCSeconds();r()[b+4>>>2>>>0]=a.getUTCMinutes();r()[b+8>>>2>>>0]=a.getUTCHours();r()[b+12>>>2>>>0]=a.getUTCDate();r()[b+16>>>2>>>0]=a.getUTCMonth();r()[b+20>>>2>>>0]=a.getUTCFullYear()-1900;r()[b+24>>>2>>>0]=a.getUTCDay();a=(a.getTime()-Date.UTC(a.getUTCFullYear(),0,1,0,0,0,0))/864E5|0;r()[b+28>>>2>>>0]=a}
var X=a=>0===a%4&&(0!==a%100||0===a%400),vc=[0,31,60,91,121,152,182,213,244,274,305,335],wc=[0,31,59,90,120,151,181,212,243,273,304,334];
function ob(a,b){a=-9007199254740992>a||9007199254740992<a?NaN:Number(a);b>>>=0;a=new Date(1E3*a);r()[b>>>2>>>0]=a.getSeconds();r()[b+4>>>2>>>0]=a.getMinutes();r()[b+8>>>2>>>0]=a.getHours();r()[b+12>>>2>>>0]=a.getDate();r()[b+16>>>2>>>0]=a.getMonth();r()[b+20>>>2>>>0]=a.getFullYear()-1900;r()[b+24>>>2>>>0]=a.getDay();var c=(X(a.getFullYear())?vc:wc)[a.getMonth()]+a.getDate()-1|0;r()[b+28>>>2>>>0]=c;r()[b+36>>>2>>>0]=-(60*a.getTimezoneOffset());c=(new Date(a.getFullYear(),6,1)).getTimezoneOffset();
var d=(new Date(a.getFullYear(),0,1)).getTimezoneOffset();a=(c!=d&&a.getTimezoneOffset()==Math.min(d,c))|0;r()[b+32>>>2>>>0]=a}
function pb(a){a>>>=0;var b=new Date(r()[a+20>>>2>>>0]+1900,r()[a+16>>>2>>>0],r()[a+12>>>2>>>0],r()[a+8>>>2>>>0],r()[a+4>>>2>>>0],r()[a>>>2>>>0],0),c=r()[a+32>>>2>>>0],d=b.getTimezoneOffset(),g=(new Date(b.getFullYear(),6,1)).getTimezoneOffset(),h=(new Date(b.getFullYear(),0,1)).getTimezoneOffset(),k=Math.min(h,g);0>c?r()[a+32>>>2>>>0]=Number(g!=h&&k==d):0<c!=(k==d)&&(g=Math.max(h,g),b.setTime(b.getTime()+6E4*((0<c?k:g)-d)));r()[a+24>>>2>>>0]=b.getDay();c=(X(b.getFullYear())?vc:wc)[b.getMonth()]+
b.getDate()-1|0;r()[a+28>>>2>>>0]=c;r()[a>>>2>>>0]=b.getSeconds();r()[a+4>>>2>>>0]=b.getMinutes();r()[a+8>>>2>>>0]=b.getHours();r()[a+12>>>2>>>0]=b.getDate();r()[a+16>>>2>>>0]=b.getMonth();r()[a+20>>>2>>>0]=b.getYear();a=b.getTime();return BigInt(isNaN(a)?-1:a/1E3)}function qb(a,b,c,d,g,h,k){return C?V(16,1,a,b,c,d,g,h,k):-52}function rb(a,b,c,d,g,h){if(C)return V(17,1,a,b,c,d,g,h)}
function sb(a,b,c,d){a>>>=0;b>>>=0;c>>>=0;d>>>=0;var g=(new Date).getFullYear(),h=new Date(g,0,1),k=new Date(g,6,1);g=h.getTimezoneOffset();var q=k.getTimezoneOffset(),z=Math.max(g,q);u()[a>>>2>>>0]=60*z;r()[b>>>2>>>0]=Number(g!=q);a=v=>v.toLocaleTimeString(void 0,{hour12:!1,timeZoneName:"short"}).split(" ")[1];h=a(h);k=a(k);q<g?(W(h,c,17),W(k,d,17)):(W(h,d,17),W(k,c,17))}var xc=[];
function tb(a,b,c){a>>>=0;b>>>=0;c>>>=0;xc.length=0;for(var d;d=n()[b++>>>0];){var g=105!=d;g&=112!=d;c+=g&&c%8?4:0;xc.push(112==d?u()[c>>>2>>>0]:106==d?N[c>>>3]:105==d?r()[c>>>2>>>0]:ea()[c>>>3>>>0]);c+=g?8:4}return Nb[a](...xc)}var ub=()=>{},vb=()=>Date.now();function wb(a,b){return K(Q(a>>>0,b>>>0))}var xb=()=>{U+=1;throw"unwind";};function yb(){return 4294901760}var zb;zb=()=>performance.timeOrigin+performance.now();var Ab=()=>B?require("os").cpus().length:navigator.hardwareConcurrency;
function Bb(a){a>>>=0;var b=n().length;if(a<=b||4294901760<a)return!1;for(var c=1;4>=c;c*=2){var d=b*(1+.2/c);d=Math.min(d,a+100663296);var g=Math;d=Math.max(a,d);a:{g=(g.min.call(g,4294901760,d+(65536-d%65536)%65536)-f.buffer.byteLength+65535)/65536;try{f.grow(g);m();var h=1;break a}catch(k){}h=void 0}if(h)return!0}return!1}var yc=()=>{Ka("Cannot use convertFrameToPC (needed by __builtin_return_address) without -sUSE_OFFSET_CONVERTER");return 0},Y={},zc=a=>{a.forEach(b=>{var c=yc();c&&(Y[c]=b)})};
function Cb(){var a=Error().stack.toString().split("\n");"Error"==a[0]&&a.shift();zc(a);Y.Xa=yc();Y.Za=a;return Y.Xa}function Db(a,b,c){a>>>=0;b>>>=0;if(Y.Xa==a)var d=Y.Za;else d=Error().stack.toString().split("\n"),"Error"==d[0]&&d.shift(),zc(d);for(var g=3;d[g]&&yc()!=a;)++g;for(a=0;a<c&&d[a+g];++a)r()[b+4*a>>>2>>>0]=yc();return a}
var Ac={},Cc=()=>{if(!Bc){var a={USER:"web_user",LOGNAME:"web_user",PATH:"/",PWD:"/",HOME:"/home/web_user",LANG:("object"==typeof navigator&&navigator.languages&&navigator.languages[0]||"C").replace("-","_")+".UTF-8",_:la||"./this.program"},b;for(b in Ac)void 0===Ac[b]?delete a[b]:a[b]=Ac[b];var c=[];for(b in a)c.push(`${b}=${a[b]}`);Bc=c}return Bc},Bc;
function Eb(a,b){if(C)return V(18,1,a,b);a>>>=0;b>>>=0;var c=0;Cc().forEach((d,g)=>{var h=b+c;g=u()[a+4*g>>>2>>>0]=h;for(h=0;h<d.length;++h)aa()[g++>>>0]=d.charCodeAt(h);aa()[g>>>0]=0;c+=d.length+1});return 0}function Fb(a,b){if(C)return V(19,1,a,b);a>>>=0;b>>>=0;var c=Cc();u()[a>>>2>>>0]=c.length;var d=0;c.forEach(g=>d+=g.length+1);u()[b>>>2>>>0]=d;return 0}function Hb(a){return C?V(20,1,a):52}function Ib(a,b,c,d){return C?V(21,1,a,b,c,d):52}function Jb(a,b,c,d){return C?V(22,1,a,b,c,d):70}
var Dc=[null,[],[]];function Kb(a,b,c,d){if(C)return V(23,1,a,b,c,d);b>>>=0;c>>>=0;d>>>=0;for(var g=0,h=0;h<c;h++){var k=u()[b>>>2>>>0],q=u()[b+4>>>2>>>0];b+=8;for(var z=0;z<q;z++){var v=n()[k+z>>>0],A=Dc[a];0===v||10===v?((1===a?qa:K)(pc(A,0)),A.length=0):A.push(v)}g+=q}u()[d>>>2>>>0]=g;return 0}var Ec=[31,29,31,30,31,30,31,31,30,31,30,31],Fc=[31,28,31,30,31,30,31,31,30,31,30,31];function Gc(a){var b=Array(qc(a)+1);rc(a,b,0,b.length);return b}var Hc=(a,b)=>{aa().set(a,b>>>0)};
function Lb(a,b,c,d){function g(e,p,t){for(e="number"==typeof e?e.toString():e||"";e.length<p;)e=t[0]+e;return e}function h(e,p){return g(e,p,"0")}function k(e,p){function t(Wb){return 0>Wb?-1:0<Wb?1:0}var J;0===(J=t(e.getFullYear()-p.getFullYear()))&&0===(J=t(e.getMonth()-p.getMonth()))&&(J=t(e.getDate()-p.getDate()));return J}function q(e){switch(e.getDay()){case 0:return new Date(e.getFullYear()-1,11,29);case 1:return e;case 2:return new Date(e.getFullYear(),0,3);case 3:return new Date(e.getFullYear(),
0,2);case 4:return new Date(e.getFullYear(),0,1);case 5:return new Date(e.getFullYear()-1,11,31);case 6:return new Date(e.getFullYear()-1,11,30)}}function z(e){var p=e.Pa;for(e=new Date((new Date(e.Qa+1900,0,1)).getTime());0<p;){var t=e.getMonth(),J=(X(e.getFullYear())?Ec:Fc)[t];if(p>J-e.getDate())p-=J-e.getDate()+1,e.setDate(1),11>t?e.setMonth(t+1):(e.setMonth(0),e.setFullYear(e.getFullYear()+1));else{e.setDate(e.getDate()+p);break}}t=new Date(e.getFullYear()+1,0,4);p=q(new Date(e.getFullYear(),
0,4));t=q(t);return 0>=k(p,e)?0>=k(t,e)?e.getFullYear()+1:e.getFullYear():e.getFullYear()-1}a>>>=0;b>>>=0;c>>>=0;d>>>=0;var v=u()[d+40>>>2>>>0];d={cb:r()[d>>>2>>>0],bb:r()[d+4>>>2>>>0],Sa:r()[d+8>>>2>>>0],Va:r()[d+12>>>2>>>0],Ta:r()[d+16>>>2>>>0],Qa:r()[d+20>>>2>>>0],Na:r()[d+24>>>2>>>0],Pa:r()[d+28>>>2>>>0],kb:r()[d+32>>>2>>>0],ab:r()[d+36>>>2>>>0],eb:v?Q(v):""};c=Q(c);v={"%c":"%a %b %d %H:%M:%S %Y","%D":"%m/%d/%y","%F":"%Y-%m-%d","%h":"%b","%r":"%I:%M:%S %p","%R":"%H:%M","%T":"%H:%M:%S","%x":"%m/%d/%y",
"%X":"%H:%M:%S","%Ec":"%c","%EC":"%C","%Ex":"%m/%d/%y","%EX":"%H:%M:%S","%Ey":"%y","%EY":"%Y","%Od":"%d","%Oe":"%e","%OH":"%H","%OI":"%I","%Om":"%m","%OM":"%M","%OS":"%S","%Ou":"%u","%OU":"%U","%OV":"%V","%Ow":"%w","%OW":"%W","%Oy":"%y"};for(var A in v)c=c.replace(new RegExp(A,"g"),v[A]);var Xb="Sunday Monday Tuesday Wednesday Thursday Friday Saturday".split(" "),Yb="January February March April May June July August September October November December".split(" ");v={"%a":e=>Xb[e.Na].substring(0,3),
"%A":e=>Xb[e.Na],"%b":e=>Yb[e.Ta].substring(0,3),"%B":e=>Yb[e.Ta],"%C":e=>h((e.Qa+1900)/100|0,2),"%d":e=>h(e.Va,2),"%e":e=>g(e.Va,2," "),"%g":e=>z(e).toString().substring(2),"%G":z,"%H":e=>h(e.Sa,2),"%I":e=>{e=e.Sa;0==e?e=12:12<e&&(e-=12);return h(e,2)},"%j":e=>{for(var p=0,t=0;t<=e.Ta-1;p+=(X(e.Qa+1900)?Ec:Fc)[t++]);return h(e.Va+p,3)},"%m":e=>h(e.Ta+1,2),"%M":e=>h(e.bb,2),"%n":()=>"\n","%p":e=>0<=e.Sa&&12>e.Sa?"AM":"PM","%S":e=>h(e.cb,2),"%t":()=>"\t","%u":e=>e.Na||7,"%U":e=>h(Math.floor((e.Pa+
7-e.Na)/7),2),"%V":e=>{var p=Math.floor((e.Pa+7-(e.Na+6)%7)/7);2>=(e.Na+371-e.Pa-2)%7&&p++;if(p)53==p&&(t=(e.Na+371-e.Pa)%7,4==t||3==t&&X(e.Qa)||(p=1));else{p=52;var t=(e.Na+7-e.Pa-1)%7;(4==t||5==t&&X(e.Qa%400-1))&&p++}return h(p,2)},"%w":e=>e.Na,"%W":e=>h(Math.floor((e.Pa+7-(e.Na+6)%7)/7),2),"%y":e=>(e.Qa+1900).toString().substring(2),"%Y":e=>e.Qa+1900,"%z":e=>{e=e.ab;var p=0<=e;e=Math.abs(e)/60;return(p?"+":"-")+String("0000"+(e/60*100+e%60)).slice(-4)},"%Z":e=>e.eb,"%%":()=>"%"};c=c.replace(/%%/g,
"\x00\x00");for(A in v)c.includes(A)&&(c=c.replace(new RegExp(A,"g"),v[A](d)));c=c.replace(/\0\0/g,"%");A=Gc(c);if(A.length>b)return 0;Hc(A,a);return A.length-1}function Mb(a,b,c,d){return Lb(a>>>0,b>>>0,c>>>0,d>>>0)}C||cc();
var uc=[$b,ac,nc,Ua,Va,Wa,Xa,Ya,Za,$a,ab,bb,cb,db,eb,fb,qb,rb,Eb,Fb,Hb,Ib,Jb,Kb],Ra,Z=function(){function a(c,d){Z=c.exports;Z=Ic();bc.push(Z.Ba);jc=Z.Ca;Ga.unshift(Z.Y);Da=d;Ja();return Z}var b=ua();O++;if(w.instantiateWasm)try{return w.instantiateWasm(b,a)}catch(c){K(`Module.instantiateWasm callback failed with error: ${c}`),x(c)}Ma||=w.locateFile?La("ort-wasm-simd-threaded.wasm")?"ort-wasm-simd-threaded.wasm":w.locateFile?w.locateFile("ort-wasm-simd-threaded.wasm",F):F+"ort-wasm-simd-threaded.wasm":
(new URL("ort-wasm-simd-threaded.wasm",import.meta.url)).href;Qa(b,function(c){a(c.instance,c.module)}).catch(x);return{}}();w._OrtInit=(a,b)=>(w._OrtInit=Z.Z)(a,b);w._OrtGetLastError=(a,b)=>(w._OrtGetLastError=Z._)(a,b);w._OrtCreateSessionOptions=(a,b,c,d,g,h,k,q,z,v)=>(w._OrtCreateSessionOptions=Z.$)(a,b,c,d,g,h,k,q,z,v);w._OrtAppendExecutionProvider=(a,b)=>(w._OrtAppendExecutionProvider=Z.aa)(a,b);w._OrtAddFreeDimensionOverride=(a,b,c)=>(w._OrtAddFreeDimensionOverride=Z.ba)(a,b,c);
w._OrtAddSessionConfigEntry=(a,b,c)=>(w._OrtAddSessionConfigEntry=Z.ca)(a,b,c);w._OrtReleaseSessionOptions=a=>(w._OrtReleaseSessionOptions=Z.da)(a);w._OrtCreateSession=(a,b,c)=>(w._OrtCreateSession=Z.ea)(a,b,c);w._OrtReleaseSession=a=>(w._OrtReleaseSession=Z.fa)(a);w._OrtGetInputOutputCount=(a,b,c)=>(w._OrtGetInputOutputCount=Z.ga)(a,b,c);w._OrtGetInputName=(a,b)=>(w._OrtGetInputName=Z.ha)(a,b);w._OrtGetOutputName=(a,b)=>(w._OrtGetOutputName=Z.ia)(a,b);w._OrtFree=a=>(w._OrtFree=Z.ja)(a);
w._OrtCreateTensor=(a,b,c,d,g,h)=>(w._OrtCreateTensor=Z.ka)(a,b,c,d,g,h);w._OrtGetTensorData=(a,b,c,d,g)=>(w._OrtGetTensorData=Z.la)(a,b,c,d,g);w._OrtReleaseTensor=a=>(w._OrtReleaseTensor=Z.ma)(a);w._OrtCreateRunOptions=(a,b,c,d)=>(w._OrtCreateRunOptions=Z.na)(a,b,c,d);w._OrtAddRunConfigEntry=(a,b,c)=>(w._OrtAddRunConfigEntry=Z.oa)(a,b,c);w._OrtReleaseRunOptions=a=>(w._OrtReleaseRunOptions=Z.pa)(a);w._OrtCreateBinding=a=>(w._OrtCreateBinding=Z.qa)(a);
w._OrtBindInput=(a,b,c)=>(w._OrtBindInput=Z.ra)(a,b,c);w._OrtBindOutput=(a,b,c,d)=>(w._OrtBindOutput=Z.sa)(a,b,c,d);w._OrtClearBoundOutputs=a=>(w._OrtClearBoundOutputs=Z.ta)(a);w._OrtReleaseBinding=a=>(w._OrtReleaseBinding=Z.ua)(a);w._OrtRunWithBinding=(a,b,c,d,g)=>(w._OrtRunWithBinding=Z.va)(a,b,c,d,g);w._OrtRun=(a,b,c,d,g,h,k,q)=>(w._OrtRun=Z.wa)(a,b,c,d,g,h,k,q);w._OrtEndProfiling=a=>(w._OrtEndProfiling=Z.xa)(a);var L=()=>(L=Z.ya)();w._malloc=a=>(w._malloc=Z.za)(a);w._free=a=>(w._free=Z.Aa)(a);
var va=(a,b,c,d,g,h)=>(va=Z.Da)(a,b,c,d,g,h),Ca=()=>(Ca=Z.Ea)(),Vb=(a,b,c,d,g)=>(Vb=Z.Fa)(a,b,c,d,g),ec=a=>(ec=Z.Ga)(a),Aa=a=>(Aa=Z.Ha)(a),sc=()=>(sc=Z.Ia)(),hc=(a,b)=>(hc=Z.Ja)(a,b),Zb=a=>(Zb=Z.Ka)(a),Ub=a=>(Ub=Z.La)(a),Tb=()=>(Tb=Z.Ma)();function Ic(){var a=Z;a=Object.assign({},a);var b=d=>()=>d()>>>0,c=d=>g=>d(g)>>>0;a.ya=b(a.ya);a.za=c(a.za);a.emscripten_main_runtime_thread_id=b(a.emscripten_main_runtime_thread_id);a.La=c(a.La);a.Ma=b(a.Ma);return a}w.stackSave=()=>Tb();w.stackRestore=a=>Zb(a);
w.stackAlloc=a=>Ub(a);w.UTF8ToString=Q;w.stringToUTF8=W;w.lengthBytesUTF8=qc;var Jc;P=function Kc(){Jc||Lc();Jc||(P=Kc)};function Lc(){if(!(0<O))if(C)ha(w),C||gc(Ga),startWorker(w);else{if(w.preRun)for("function"==typeof w.preRun&&(w.preRun=[w.preRun]);w.preRun.length;)Fa.unshift(w.preRun.shift());gc(Fa);0<O||Jc||(Jc=!0,w.calledRun=!0,Ea||(C||gc(Ga),ha(w),C||gc(Ha)))}}Lc();moduleRtn=ia;


  return moduleRtn;
}
);
})();
export default ortWasmThreaded;
var isPthread = globalThis.self?.name === 'em-pthread';
var isNode = typeof globalThis.process?.versions?.node == 'string';
if (isNode) isPthread = (await import('worker_threads')).workerData === 'em-pthread';

// When running as a pthread, construct a new instance on startup
isPthread && ortWasmThreaded();
