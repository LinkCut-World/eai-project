import { getDocument, GlobalWorkerOptions } from './vendor/pdfjs/pdf.min.mjs';

GlobalWorkerOptions.workerSrc = new URL('./vendor/pdfjs/pdf.worker.min.mjs', import.meta.url).href;

const stage = document.querySelector('#report-stage');
const status = document.querySelector('#report-status');
const pageLabel = document.querySelector('#page-label');
const zoomLabel = document.querySelector('#zoom-label');
const text = document.querySelector('#report-text');
const previous = document.querySelector('#previous-page');
const next = document.querySelector('#next-page');
const zoomOut = document.querySelector('#zoom-out');
const zoomIn = document.querySelector('#zoom-in');
let pdf;
let pageNumber = 1;
let zoom = 1;
let generation = 0;
let rendering;

function updateControls() {
  pageLabel.textContent = `Page ${pageNumber} of ${pdf.numPages}`;
  zoomLabel.textContent = `${Math.round(zoom * 100)}%`;
  previous.disabled = pageNumber === 1;
  next.disabled = pageNumber === pdf.numPages;
  zoomOut.disabled = zoom <= 0.75;
  zoomIn.disabled = zoom >= 2;
}

async function renderPage() {
  const current = ++generation;
  rendering?.cancel();
  updateControls();
  try {
    const page = await pdf.getPage(pageNumber);
    if (current !== generation) return;
    const padding = parseFloat(getComputedStyle(stage).paddingLeft) * 2;
    const available = Math.max(240, stage.clientWidth - padding - 2);
    const scale = Math.min(available, 1000) / page.getViewport({ scale: 1 }).width * zoom;
    const viewport = page.getViewport({ scale });
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    const canvas = document.createElement('canvas');
    canvas.width = Math.ceil(viewport.width * ratio);
    canvas.height = Math.ceil(viewport.height * ratio);
    canvas.style.width = `${viewport.width}px`;
    canvas.style.height = `${viewport.height}px`;
    canvas.setAttribute('aria-hidden', 'true');
    rendering = page.render({
      canvasContext: canvas.getContext('2d'),
      viewport,
      transform: ratio === 1 ? null : [ratio, 0, 0, ratio, 0, 0],
    });
    await rendering.promise;
    const content = await page.getTextContent();
    if (current !== generation) return;
    stage.replaceChildren(canvas);
    stage.scrollTop = 0;
    stage.scrollLeft = 0;
    stage.setAttribute('aria-label', `Report page ${pageNumber} of ${pdf.numPages}. Use the buttons above to change pages.`);
    text.textContent = content.items.map(item => item.str || '').join(' ');
  } catch (error) {
    if (current !== generation || error.name === 'RenderingCancelledException') return;
    console.error('Could not render the report page:', error);
    status.textContent = 'The report could not be displayed here. Use “Open PDF” above to read the complete report.';
    stage.replaceChildren(status);
  }
}

previous.addEventListener('click', () => { if (pageNumber > 1) { pageNumber--; renderPage(); } });
next.addEventListener('click', () => { if (pageNumber < pdf.numPages) { pageNumber++; renderPage(); } });
zoomOut.addEventListener('click', () => { zoom = Math.max(0.75, zoom - 0.25); renderPage(); });
zoomIn.addEventListener('click', () => { zoom = Math.min(2, zoom + 0.25); renderPage(); });

let resizeTimer;
let previousWidth = stage.clientWidth;
new ResizeObserver(() => {
  if (stage.clientWidth === previousWidth) return;
  previousWidth = stage.clientWidth;
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => { if (pdf) renderPage(); }, 150);
}).observe(stage);

async function loadReport() {
  try {
    pdf = await getDocument({
      url: new URL('./assets/report.pdf', import.meta.url).href,
      standardFontDataUrl: new URL('./vendor/pdfjs/standard_fonts/', import.meta.url).href,
      wasmUrl: new URL('./vendor/pdfjs/wasm/', import.meta.url).href,
    }).promise;
    await renderPage();
  } catch (error) {
    console.error('Could not load the report:', error);
    pageLabel.textContent = 'Project report';
    status.textContent = 'The report could not be loaded here. Use “Open PDF” above to read the complete report.';
  }
}

const observer = new IntersectionObserver(entries => {
  if (!entries.some(entry => entry.isIntersecting)) return;
  observer.disconnect();
  loadReport();
}, { rootMargin: '300px' });
observer.observe(stage);
