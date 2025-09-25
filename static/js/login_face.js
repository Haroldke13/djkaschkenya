const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const verifyBtn = document.getElementById("verifyBtn");

if (navigator.mediaDevices.getUserMedia) {
  navigator.mediaDevices.getUserMedia({ video: true })
    .then(stream => { video.srcObject = stream; })
    .catch(err => console.error("Camera error:", err));
}

verifyBtn.addEventListener("click", async () => {
  const ctx = canvas.getContext("2d");
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  const img = canvas.toDataURL("image/png");

  const res = await fetch(`/verify_face/${USER_ID}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ image: img })
  });

  const data = await res.json();
  if (res.ok) {
    window.location.href = "/profile";
  } else {
    alert("Face verification failed: " + data.message);
  }
});
