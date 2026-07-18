const fs = require("fs");
const path = require("path");
const { pathToFileURL } = require("url");

async function main() {
  const [pdfPath, outDir] = process.argv.slice(2);
  if (!pdfPath || !outDir) {
    throw new Error("Usage: node rasterize_pdf.js <input.pdf> <output_dir>");
  }
  fs.mkdirSync(outDir, { recursive: true });
  const { createCanvas } = require("@napi-rs/canvas");
  const pdfjsPath = require.resolve("pdfjs-dist/legacy/build/pdf.mjs");
  const pdfjs = await import(pathToFileURL(pdfjsPath).href);
  const data = new Uint8Array(fs.readFileSync(pdfPath));
  const loadingTask = pdfjs.getDocument({
    data,
    disableWorker: true,
    useSystemFonts: true,
  });
  const pdf = await loadingTask.promise;
  for (let pageNumber = 1; pageNumber <= pdf.numPages; pageNumber += 1) {
    const page = await pdf.getPage(pageNumber);
    const viewport = page.getViewport({ scale: 2 });
    const canvas = createCanvas(Math.ceil(viewport.width), Math.ceil(viewport.height));
    const context = canvas.getContext("2d");
    context.fillStyle = "#ffffff";
    context.fillRect(0, 0, canvas.width, canvas.height);
    await page.render({ canvasContext: context, viewport }).promise;
    const outPath = path.join(outDir, `page-${String(pageNumber).padStart(2, "0")}.png`);
    fs.writeFileSync(outPath, canvas.toBuffer("image/png"));
  }
  console.log(`pages=${pdf.numPages}`);
}

main().catch((error) => {
  console.error(error.stack || error.message);
  process.exit(1);
});
