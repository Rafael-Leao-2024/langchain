import fitz  # PyMuPDF

# Abrir o PDF
doc = fitz.open("servico.pdf")



for pagina in doc:
    # Procurar pelo texto exato
    areas = pagina.search_for("R$ 350,00")
    for area in areas:
        # Apagar o texto antigo
        pagina.add_redact_annot(area, fill=(1, 1, 1))  # fundo branco
        pagina.apply_redactions()

        # Inserir o novo texto no centro da área
        x0, y0, x1, y1 = area
        largura = x1 - x0
        altura = y1 - y0

        # Inserir texto centralizado na mesma área
        pagina.insert_text(
            (x0 + largura / 2, y0 + altura / 2),
            "R$ 400,00",
            fontsize=10,
            color=(0, 0, 0),
            fontname="helv",
            
        )

# Salvar o novo PDF
doc.save("pdf_editado.pdf")
doc.close()
