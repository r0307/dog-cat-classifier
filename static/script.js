const fileInput=document.getElementById("file-input");
const imagePreview=document.getElementById("image-preview");
const predictButton=document.getElementById("predict-button");
const resultDiv=document.getElementById("prediction-result");
const loader=document.getElementById("loader");

let selectedFile=null;

fileInput.addEventListener("change", (e)=>{
  const file=e.target.files[0];
  if(!file) return;

  selectedFile=file;

  const reader=new FileReader();
  reader.onload=(e)=>{
    imagePreview.src=e.target.result;
    imagePreview.style.display="block";
    predictButton.classList.add("visible")
    predictButton.disabled = false;
    resultDiv.innerText=" ";
  };
  reader.readAsDataURL(file);
});

predictButton.addEventListener("click", ()=>{
  if(!selectedFile){
    alert("画像をアップロードしてください。");
    return;
  }

  const formData=new FormData();
  formData.append("file", selectedFile);

  predictButton.disabled=true;
  loader.style.display="block";
  resultDiv.innerText=" ";

  fetch("./predict",{
      method:"POST",
      body:formData
  })
  .then(response => {
    if (!response.ok) {
      return response.json().then(errorData => {
        throw new Error(errorData.error||"サーバーエラーが発生しました。");
      }).catch(()=>{
        throw new Error("サーバーエラーが発生しました。");
      });
    }
    return response.json();
  })
  .then(data => {
        loader.style.display = "none";
        predictButton.disabled = false;
        
        if (data.error) {
            resultDiv.innerText = "エラー: " + data.error;
        } else {
            const confidencePercent = (data.confidence * 100).toFixed(2);
            resultDiv.innerHTML = `<span style="color:${data.prediction === "犬" ? "#3498db" : "#e74c3c"};">${data.prediction}</span>です。`;
            resultDiv.innerHTML += `<div class="result-text">確信度: ${confidencePercent}%</div>`;
        }
    })
  .catch(error=>{
        loader.style.display="none";
        predictButton.disabled=false;
        console.error("Error: ",error);
        resultDiv.innerText = "予測に失敗しました。 " + error.message;
    });
});