const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const captureBtn = document.getElementById("captureBtn");

if (navigator.mediaDevices.getUserMedia) {
  navigator.mediaDevices.getUserMedia({ video: true })
    .then(stream => { video.srcObject = stream; })
    .catch(err => console.error("Camera error:", err));
}

captureBtn?.addEventListener("click", async () => {
  const ctx = canvas.getContext("2d");
  let images = [];

  for (let i = 0; i < 10; i++) {
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    images.push(canvas.toDataURL("image/png"));
    await new Promise(r => setTimeout(r, 500)); // capture every 0.5s
  }

  const res = await fetch("/capture_face", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ images })
  });

  const data = await res.json();
  if (res.ok) {
    Swal.fire("Success", data.message, "success");
  } else {
    Swal.fire("Error", data.message, "error");
  }
});
