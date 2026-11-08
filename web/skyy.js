// Minimal browser client for the step 7 bot (07_voice_webrtc.py), in place of Pipecat's
// prebuilt /client page.
//
// It speaks the same protocol as the prebuilt client: a WebRTC offer to POST /api/offer,
// transceivers in the order the server expects (0 audio = mic, 1 video, 2 screen video),
// and a data channel. Over the data channel it sends "ping" every second (the server uses
// it to notice a disconnect) and the RTVI "client-ready" message; the bot's RTVI events
// ("user-started-speaking", "bot-started-speaking", ...) drive the status line and freeze
// Finn from the moment the visitor starts talking until Skyy answers, while Whisper
// works on the same GPU (see finn3d.js). If no answer comes (side talk is ignored), Finn
// moves again RESUME_SECS after the visitor stops.
//
// The mic keeps the browser's echo cancellation, noise suppression and auto gain, which
// is why step 7 works on speakers. Skyy's voice plays in the <audio> element and also
// goes to Finn.attach() for the mouth. If the connection drops (server restarted, laptop
// asleep) the page reconnects by itself.
//
// ?demo shows Finn without the bot: it loops demo.wav (Skyy's voices in English, Spanish
// and French) and never connects. Serve the folder with any web server, for example
// python -m http.server 8000 --directory web, then open http://localhost:8000/?demo
(() => {
  const RETRY_SECS = 3;
  const RESUME_SECS = 6;
  const status = document.getElementById("status");
  const start = document.getElementById("start");
  const voice = document.getElementById("voice");

  let pc = null;
  let pinger = null;
  let botSpeaking = false;
  let userSpeaking = false;
  let resumeTimer = null;

  function freeze(on) {
    clearTimeout(resumeTimer);
    Finn.pause(on);
  }

  function show(text) { status.textContent = text; }

  function showTurn() {
    if (botSpeaking) show("Skyy is speaking");
    else if (userSpeaking) show("Listening…");
    else show('Say "Sky" and ask me anything');
  }

  function onMessage(event) {
    if (typeof event.data !== "string" || !event.data.startsWith("{")) return;
    let msg;
    try { msg = JSON.parse(event.data); } catch { return; }
    switch (msg.type) {
      case "bot-ready": showTurn(); break;
      case "user-started-speaking": userSpeaking = true; freeze(true); showTurn(); break;
      case "user-stopped-speaking":
        userSpeaking = false;
        clearTimeout(resumeTimer);
        resumeTimer = setTimeout(() => Finn.pause(false), RESUME_SECS * 1000);
        showTurn();
        break;
      case "bot-started-speaking": botSpeaking = true; freeze(false); showTurn(); break;
      case "bot-stopped-speaking": botSpeaking = false; showTurn(); break;
    }
  }

  function waitForIce(peer) {
    // The bot runs on this laptop, so host candidates are enough: wait until they are
    // all in the offer instead of trickling them with PATCH /api/offer.
    if (peer.iceGatheringState === "complete") return Promise.resolve();
    return new Promise(resolve => {
      const done = () => {
        if (peer.iceGatheringState === "complete") {
          peer.removeEventListener("icegatheringstatechange", done);
          resolve();
        }
      };
      peer.addEventListener("icegatheringstatechange", done);
      setTimeout(resolve, 2000);
    });
  }

  function cleanup() {
    clearInterval(pinger);
    pinger = null;
    if (pc) {
      pc.getSenders().forEach(s => s.track && s.track.stop());
      pc.close();
      pc = null;
    }
    botSpeaking = userSpeaking = false;
    freeze(false);
  }

  function retry(reason) {
    console.warn("Skyy connection:", reason);
    cleanup();
    show("Reconnecting…");
    setTimeout(connect, RETRY_SECS * 1000);
  }

  async function connect() {
    show("Connecting…");
    try {
      const mic = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true }});

      const peer = pc = new RTCPeerConnection();
      peer.addTransceiver(mic.getAudioTracks()[0], { direction: "sendrecv" });
      peer.addTransceiver("video", { direction: "sendrecv" });   // camera (unused)
      peer.addTransceiver("video", { direction: "sendrecv" });   // screen share (unused)

      peer.ontrack = event => {
        if (event.track.kind !== "audio") return;
        const stream = event.streams[0] || new MediaStream([event.track]);
        voice.srcObject = stream;
        voice.play().catch(() => {});
        Finn.attach(stream);
      };

      const channel = peer.createDataChannel("chat", { ordered: true });
      channel.onopen = () => {
        pinger = setInterval(() => {
          if (channel.readyState === "open") channel.send(`ping: ${Date.now()}`);
        }, 1000);
        channel.send(JSON.stringify({
          id: crypto.randomUUID(), label: "rtvi-ai", type: "client-ready",
          data: { version: "1.0.0", about: { library: "skyy-finn", library_version: "1" } }}));
      };
      channel.onmessage = onMessage;

      peer.onconnectionstatechange = () => {
        if (peer !== pc) return;
        if (["failed", "closed", "disconnected"].includes(peer.connectionState)) {
          retry(peer.connectionState);
        }
      };

      await peer.setLocalDescription(await peer.createOffer());
      await waitForIce(peer);
      const res = await fetch("/api/offer", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sdp: peer.localDescription.sdp, type: peer.localDescription.type })});
      if (!res.ok) throw new Error(`offer rejected: HTTP ${res.status}`);
      const answer = await res.json();
      await peer.setRemoteDescription({ sdp: answer.sdp, type: answer.type });
    } catch (err) {
      retry(err);
    }
  }

  // Browsers only play sound after a click on the page, unless Chrome runs with
  // --autoplay-policy=no-user-gesture-required (kiosk). Try to start right away and
  // show the button only if the browser says no.
  async function begin() {
    start.hidden = true;
    if (demo) return playDemo();
    await connect();
  }

  function playDemo() {
    voice.src = "demo.wav";
    voice.loop = true;
    Finn.attach(voice);
    voice.play();
    show("Demo: recorded voice, no bot connected");
  }

  const demo = new URLSearchParams(location.search).has("demo");
  if (demo) start.textContent = "Play demo";

  start.onclick = begin;
  const probe = new AudioContext();
  if (probe.state === "running") {
    probe.close();
    begin();
  } else {
    probe.close();
    start.hidden = false;
    show("Click to start");
  }
})();
