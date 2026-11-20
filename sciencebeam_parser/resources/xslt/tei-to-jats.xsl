<?xml version="1.0" encoding="UTF-8"?>
<xsl:stylesheet 
  xmlns:xsl="http://www.w3.org/1999/XSL/Transform" 
  xmlns:xs="http://www.w3.org/2001/XMLSchema" 
  xmlns:mml="http://www.w3.org/1998/Math/MathML" 
  xmlns:xlink="http://www.w3.org/1999/xlink" 
  xmlns:tei="http://www.tei-c.org/ns/1.0"
  xmlns:exsl="http://exslt.org/common"
  xmlns:cite="urn:sciencebeam-parser:tei-to-jats:citation"
  exclude-result-prefixes="xs mml tei exsl cite"
  version="1.0"
>
  <xsl:output
    method="xml"
    encoding="UTF-8"
    doctype-public="-//NLM//DTD JATS (Z39.96) Journal Archiving and Interchange DTD v1.4 20241031//EN"
    doctype-system="https://jats.nlm.nih.gov/archiving/1.4/JATS-archivearticle1-4.dtd"
  />
  <xsl:param name="output_parameters" select="'false'"/>
  <xsl:param name="output_bold" select="'false'"/>
  <xsl:param name="output_italic" select="'false'"/>
  <xsl:param name="output_empty_figure_graphic" select="'false'"/>
  <xsl:param name="acknowledgement_target" select="'ack'"/>
  <xsl:param name="annex_target" select="'back'"/>

  <!-- the language the document's text is written in -->
  <xsl:variable name="language" select="string(/tei:TEI/tei:text/@xml:lang)"/>

  <xsl:template match="/">
    <article dtd-version="1.4">
      <xsl:if test="$language">
        <xsl:attribute name="xml:lang"><xsl:value-of select="$language"/></xsl:attribute>
      </xsl:if>
      <processing-meta tagset-family="jats" base-tagset="archiving">
        <xsl:apply-templates select="tei:TEI/tei:teiHeader" mode="conversion-details"/>
      </processing-meta>
      <front>
        <xsl:apply-templates select="tei:TEI/tei:teiHeader"/>
      </front>
      <body>
        <xsl:apply-templates select="tei:TEI/tei:text/tei:body"/>
        <xsl:if test="$acknowledgement_target = 'body'">
          <xsl:apply-templates select="tei:TEI/tei:text/tei:back/tei:div[@type='acknowledgement']/tei:div
            | tei:TEI/tei:text/tei:back/tei:div[@type='acknowledgement'][not(tei:div)]"/>
        </xsl:if>
        <xsl:if test="$annex_target = 'body'">
          <xsl:for-each select="tei:TEI/tei:text/tei:back/tei:div[@type='annex']">
            <xsl:apply-templates select="tei:div"/>
            <xsl:if test="tei:figure">
              <sec id="annex_figures">
                <title>Annex Figures</title>
                <xsl:apply-templates select="tei:figure"/>
              </sec>
            </xsl:if>
          </xsl:for-each>
        </xsl:if>
      </body>
      <back>
        <xsl:apply-templates select="tei:TEI/tei:text/tei:back"/>
      </back>
    </article>
  </xsl:template>

  <xsl:template match="tei:teiHeader">
    <xsl:if test="tei:fileDesc/tei:sourceDesc/tei:biblStruct/tei:monogr/tei:title">
      <journal-meta>
        <journal-title-group>
          <journal-title>
            <xsl:value-of select="tei:fileDesc/tei:sourceDesc/tei:biblStruct/tei:monogr/tei:title"/>
          </journal-title>
        </journal-title-group>
      </journal-meta>
    </xsl:if>

    <article-meta>
      <xsl:if test="not(tei:fileDesc/tei:titleStmt/tei:title)">
        <title-group><article-title/></title-group>
      </xsl:if>
      <xsl:for-each select="tei:fileDesc/tei:titleStmt/tei:title">
        <title-group>
          <xsl:call-template name="language-unless-the-document-s"/>
          <article-title>
            <xsl:apply-templates select="."/>
          </article-title>
        </title-group>
      </xsl:for-each>

      <xsl:if test="tei:fileDesc/tei:sourceDesc/tei:biblStruct/tei:analytic/tei:author">
        <contrib-group>
          <xsl:for-each select="tei:fileDesc/tei:sourceDesc/tei:biblStruct/tei:analytic/tei:author">
            <contrib contrib-type="author">
              <xsl:apply-templates select="tei:persName"/>

              <xsl:if test="tei:email">
                <email>
                  <xsl:value-of select="tei:email"/>
                </email>
              </xsl:if>

              <!-- one cross reference per affiliation attached to the author -->
              <xsl:for-each select="tei:affiliation">
                <xref ref-type="aff">
                  <xsl:attribute name='rid'>
                    <xsl:value-of select="@key"/>
                  </xsl:attribute>
                  <xsl:value-of select="tei:note[@type='raw_affiliation']/tei:label"/>
                </xref>
              </xsl:for-each>
            </contrib>
          </xsl:for-each>

          <!-- an affiliation attached to several authors is written out under each of
               them in the TEI; the JATS carries it once, under its key -->
          <xsl:for-each select="tei:fileDesc/tei:sourceDesc/tei:biblStruct/tei:analytic/tei:author/tei:affiliation[not(@key = preceding::tei:affiliation/@key)]">
            <aff>
              <xsl:attribute name='id'>
                <xsl:value-of select="@key"/>
              </xsl:attribute>
              <xsl:variable name="raw" select="tei:note[@type='raw_affiliation']"/>
              <xsl:variable name="parts" select="tei:orgName | tei:address/*[not(self::tei:postBox)]"/>
              <xsl:if test="$raw/tei:label">
                <label><xsl:value-of select="$raw/tei:label"/></label>
              </xsl:if>
              <xsl:choose>
                <xsl:when test="$raw">
                  <xsl:variable name="raw_text">
                    <xsl:for-each select="$raw/node()[not(self::tei:label)]">
                      <xsl:value-of select="."/>
                    </xsl:for-each>
                  </xsl:variable>
                  <xsl:call-template name="aff-mixed-content">
                    <xsl:with-param name="rest" select="string($raw_text)"/>
                    <xsl:with-param name="parts" select="$parts"/>
                  </xsl:call-template>
                </xsl:when>
                <xsl:otherwise>
                  <xsl:for-each select="$parts">
                    <xsl:if test="position() &gt; 1">
                      <xsl:text>, </xsl:text>
                    </xsl:if>
                    <xsl:apply-templates select="." mode="aff-part"/>
                  </xsl:for-each>
                </xsl:otherwise>
              </xsl:choose>
            </aff>
          </xsl:for-each>
        </contrib-group>
      </xsl:if>

      <xsl:if test="not(tei:profileDesc/tei:abstract)">
        <abstract/>
      </xsl:if>
      <xsl:for-each select="tei:profileDesc/tei:abstract">
        <abstract>
          <xsl:call-template name="language-unless-the-document-s"/>
          <!-- The heading the page printed above this abstract, which JATS puts
               inside it as a title. -->
          <xsl:if test="tei:div/tei:head">
            <title>
              <xsl:value-of select="tei:div/tei:head"/>
            </title>
          </xsl:if>
          <xsl:apply-templates select="tei:div/tei:p"/>
        </abstract>
      </xsl:for-each>
    </article-meta>
  </xsl:template>

  <!-- a part's own language, where it is not the one the document is written in -->
  <xsl:template name="language-unless-the-document-s">
    <xsl:if test="@xml:lang and @xml:lang != $language">
      <xsl:copy-of select="@xml:lang"/>
    </xsl:if>
  </xsl:template>

  <xsl:template match="tei:teiHeader" mode="conversion-details">
    <!-- JATS allows one custom-meta-group, so attribution and the parameters
         share it. The attribution is read from the TEI rather than passed in,
         so the two outputs cannot disagree about what produced them. -->
    <xsl:if test="tei:encodingDesc/tei:appInfo/tei:application or $output_parameters = 'true'">
      <custom-meta-group>
        <xsl:for-each select="tei:encodingDesc/tei:appInfo/tei:application">
          <custom-meta>
            <meta-name>sciencebeam-parser-version</meta-name>
            <meta-value><xsl:value-of select="@version"/></meta-value>
          </custom-meta>
          <xsl:for-each select="tei:label[@type='profile']">
            <custom-meta>
              <meta-name>sciencebeam-parser-profile</meta-name>
              <meta-value><xsl:value-of select="."/></meta-value>
            </custom-meta>
          </xsl:for-each>
          <xsl:for-each select="tei:label[@type='profile-digest']">
            <custom-meta>
              <meta-name>sciencebeam-parser-profile-digest</meta-name>
              <meta-value><xsl:value-of select="."/></meta-value>
            </custom-meta>
          </xsl:for-each>
        </xsl:for-each>
        <xsl:if test="$output_parameters = 'true'">
          <custom-meta>
            <meta-name>xslt-param-acknowledgement_target</meta-name>
            <meta-value><xsl:value-of select="$acknowledgement_target"/></meta-value>
          </custom-meta>
          <custom-meta>
            <meta-name>xslt-param-annex_target</meta-name>
            <meta-value><xsl:value-of select="$annex_target"/></meta-value>
          </custom-meta>
        </xsl:if>
      </custom-meta-group>
    </xsl:if>
  </xsl:template>

  <xsl:template match="tei:body">
    <xsl:apply-templates select="tei:div"/>
    <xsl:if test="tei:figure">
      <sec id="figures">
        <title>Figures</title>
        <xsl:apply-templates select="tei:figure"/>
      </sec>
    </xsl:if>
  </xsl:template>

  <xsl:template match="tei:graphic[normalize-space(@url)]">
    <graphic xlink:href="{@url}"/>
  </xsl:template>

  <!-- a graphic whose file was not kept has nothing for xlink:href to name -->
  <xsl:template match="tei:graphic"/>

  <xsl:template match="tei:figure[not(@type='table')]">
    <fig>
      <xsl:attribute name='id'>
        <xsl:value-of select="@xml:id"/>
      </xsl:attribute>
      <object-id><xsl:value-of select="@xml:id"/></object-id>
      <xsl:call-template name="figure-label"/>
      <caption>
        <p><xsl:apply-templates select="tei:figDesc"/></p>
      </caption>
      <xsl:apply-templates select="tei:graphic"/>
      <xsl:if test="$output_empty_figure_graphic = 'true'">
        <xsl:if test="not(tei:graphic[normalize-space(@url)])">
          <graphic/>
        </xsl:if>
      </xsl:if>
    </fig>
  </xsl:template>

  <xsl:template match="tei:figure[@type='table']">
    <table-wrap>
      <xsl:attribute name='id'>
        <xsl:value-of select="@xml:id"/>
      </xsl:attribute>
      <xsl:call-template name="figure-label"/>
      <caption>
        <p><xsl:apply-templates select="tei:figDesc"/></p>
      </caption>
      <table>
        <tbody>
          <tr>
            <td>
              <xsl:value-of select="tei:table"/>
            </td>
          </tr>
        </tbody>
      </table>
      <xsl:apply-templates select="tei:graphic"/>
    </table-wrap>
  </xsl:template>

  <!-- the label as printed: the number with the words around it, such as "Figure 3." -->
  <xsl:template name="figure-label">
    <xsl:variable name="parts" select="tei:label | tei:note[@type='&lt;figure_head&gt;']"/>
    <xsl:if test="normalize-space($parts)">
      <label>
        <xsl:for-each select="$parts[normalize-space()]">
          <xsl:if test="position() &gt; 1 and not(contains('.,:;)', substring(normalize-space(), 1, 1)))">
            <xsl:text> </xsl:text>
          </xsl:if>
          <xsl:value-of select="normalize-space()"/>
        </xsl:for-each>
      </label>
    </xsl:if>
  </xsl:template>

  <xsl:template match="tei:div">
    <sec>
      <xsl:if test="normalize-space(tei:head/@n)">
        <label><xsl:value-of select="normalize-space(tei:head/@n)"/></label>
      </xsl:if>
      <xsl:apply-templates select="tei:head"/>
      <xsl:apply-templates select="tei:p"/>
    </sec>
  </xsl:template>

  <xsl:template match="tei:back">
    <xsl:if test="$acknowledgement_target = 'ack'">
      <xsl:if test="tei:div[@type='acknowledgement']">
        <ack>
          <xsl:apply-templates select="tei:div[@type='acknowledgement']/tei:div
            | tei:div[@type='acknowledgement'][not(tei:div)]"/>
        </ack>
      </xsl:if>
    </xsl:if>
    <xsl:if test="$annex_target = 'back'">
      <xsl:for-each select="tei:div[@type='annex']">
        <xsl:apply-templates select="tei:div"/>
        <xsl:if test="tei:figure">
          <sec id="annex_figures">
            <title>Annex Figures</title>
            <xsl:apply-templates select="tei:figure"/>
          </sec>
        </xsl:if>
      </xsl:for-each>
    </xsl:if>
    <xsl:apply-templates select="tei:div/tei:listBibl"/>
    <xsl:if test="$annex_target = 'app'">
      <xsl:if test="tei:div[@type='annex']">
        <app-group>
          <app id="appendix-1">
            <title>Appendix 1</title>
            <!-- an appendix holds its figures before its sections -->
            <xsl:apply-templates select="tei:div[@type='annex']/tei:figure"/>
            <xsl:apply-templates select="tei:div[@type='annex']/tei:div"/>
          </app>
        </app-group>
      </xsl:if>
    </xsl:if>
  </xsl:template>

  <!-- the printed affiliation, each part tagged where its text is found; text no part
       accounts for stays as printed -->
  <xsl:template name="aff-mixed-content">
    <xsl:param name="rest"/>
    <xsl:param name="parts"/>
    <xsl:param name="placed" select="'|'"/>
    <xsl:variable name="candidates" select="$parts[
      normalize-space(.)
      and contains($rest, string(.))
      and not(contains($placed, concat('|', generate-id(), '|')))
    ]"/>
    <xsl:choose>
      <xsl:when test="$candidates">
        <xsl:for-each select="$candidates">
          <xsl:sort select="string-length(substring-before($rest, string(.)))" data-type="number"/>
          <xsl:sort select="string-length(.)" data-type="number" order="descending"/>
          <xsl:if test="position() = 1">
            <xsl:value-of select="substring-before($rest, string(.))"/>
            <xsl:apply-templates select="." mode="aff-part"/>
            <xsl:call-template name="aff-mixed-content">
              <xsl:with-param name="rest" select="substring-after($rest, string(.))"/>
              <xsl:with-param name="parts" select="$parts"/>
              <xsl:with-param name="placed" select="concat($placed, generate-id(), '|')"/>
            </xsl:call-template>
          </xsl:if>
        </xsl:for-each>
      </xsl:when>
      <xsl:otherwise>
        <xsl:value-of select="$rest"/>
      </xsl:otherwise>
    </xsl:choose>
  </xsl:template>

  <xsl:template match="tei:orgName[@type='institution']" mode="aff-part">
    <institution content-type="orgname"><xsl:value-of select="."/></institution>
  </xsl:template>

  <xsl:template match="tei:orgName[@type='department']" mode="aff-part">
    <institution content-type="orgdiv1"><xsl:value-of select="."/></institution>
  </xsl:template>

  <xsl:template match="tei:orgName[@type='laboratory']" mode="aff-part">
    <institution content-type="orgdiv2"><xsl:value-of select="."/></institution>
  </xsl:template>

  <xsl:template match="tei:orgName" mode="aff-part">
    <institution><xsl:value-of select="."/></institution>
  </xsl:template>

  <xsl:template match="tei:settlement" mode="aff-part">
    <city><xsl:value-of select="."/></city>
  </xsl:template>

  <xsl:template match="tei:region" mode="aff-part">
    <state><xsl:value-of select="."/></state>
  </xsl:template>

  <xsl:template match="tei:postCode" mode="aff-part">
    <postal-code><xsl:value-of select="."/></postal-code>
  </xsl:template>

  <xsl:template match="tei:addrLine" mode="aff-part">
    <addr-line><xsl:value-of select="."/></addr-line>
  </xsl:template>

  <xsl:template match="tei:country" mode="aff-part">
    <country><xsl:value-of select="."/></country>
  </xsl:template>

  <xsl:template match="tei:listBibl">
    <xsl:if test="tei:biblStruct">
      <ref-list id="ref-list-1">
        <xsl:apply-templates select="tei:biblStruct"/>
      </ref-list>
    </xsl:if>
  </xsl:template>

  <xsl:template match="tei:biblStruct">
    <ref>
      <xsl:attribute name='id'>
        <xsl:value-of select="./@xml:id"/>
      </xsl:attribute>

      <xsl:if test="normalize-space(tei:note[@type='label'])">
        <label><xsl:value-of select="normalize-space(tei:note[@type='label'])"/></label>
      </xsl:if>

      <xsl:variable name="element-citation">
        <xsl:call-template name="element-citation"/>
      </xsl:variable>
      <xsl:choose>
        <xsl:when test="normalize-space(tei:note[@type='raw_reference'])">
          <xsl:call-template name="mixed-citation"/>
        </xsl:when>
        <xsl:when test="exsl:node-set($element-citation)/*/node()">
          <xsl:copy-of select="$element-citation"/>
        </xsl:when>
        <xsl:otherwise>
          <mixed-citation publication-type="journal"/>
        </xsl:otherwise>
      </xsl:choose>
    </ref>
  </xsl:template>

  <!-- the parsed parts alone, where the printed reference is not known -->
  <xsl:template name="element-citation">
    <element-citation publication-type="journal">
      <xsl:if test="tei:analytic/tei:author/tei:persName">
        <person-group person-group-type="author">
          <xsl:apply-templates select="tei:analytic/tei:author/tei:persName"/>
        </person-group>
      </xsl:if>

      <xsl:if test="tei:monogr/tei:author/tei:persName">
        <person-group person-group-type="author">
          <xsl:apply-templates select="tei:monogr/tei:author/tei:persName"/>
        </person-group>
      </xsl:if>

      <xsl:choose>
        <xsl:when test="tei:analytic/tei:title[@type='main']">
          <article-title>
            <xsl:value-of select="tei:analytic/tei:title[@type='main']"/>
          </article-title>
        </xsl:when>
        <xsl:when test="tei:monogr/tei:title[@type='main']">
          <article-title>
            <xsl:value-of select="tei:monogr/tei:title[@type='main']"/>
          </article-title>
        </xsl:when>
      </xsl:choose>

      <xsl:if test="tei:monogr/tei:title[@level='j']">
        <source>
          <xsl:value-of select="tei:monogr/tei:title[@level='j']"/>
        </source>
      </xsl:if>

      <xsl:if test="tei:monogr/tei:imprint/tei:date[@type='published']">
        <xsl:call-template name="parseDateComponents">
          <xsl:with-param name="date" select="tei:monogr/tei:imprint/tei:date[@type='published']/@when"/>
        </xsl:call-template>
      </xsl:if>

      <xsl:if test="tei:monogr/tei:imprint/tei:biblScope[@unit='volume']">
        <volume>
          <xsl:value-of select="tei:monogr/tei:imprint/tei:biblScope[@unit='volume']"/>
        </volume>
      </xsl:if>

      <xsl:if test="tei:monogr/tei:imprint/tei:biblScope[@unit='issue']">
        <issue>
          <xsl:value-of select="tei:monogr/tei:imprint/tei:biblScope[@unit='issue']"/>
        </issue>
      </xsl:if>

      <xsl:choose>
        <xsl:when test="tei:monogr/tei:imprint/tei:biblScope[@unit='page'][@from or @to]">
          <xsl:if test="tei:monogr/tei:imprint/tei:biblScope[@unit='page']/@from">
            <fpage>
              <xsl:value-of select="tei:monogr/tei:imprint/tei:biblScope[@unit='page']/@from"/>
            </fpage>
          </xsl:if>

          <xsl:if test="tei:monogr/tei:imprint/tei:biblScope[@unit='page']/@to">
            <lpage>
              <xsl:value-of select="tei:monogr/tei:imprint/tei:biblScope[@unit='page']/@to"/>
            </lpage>
          </xsl:if>
        </xsl:when>
        <xsl:when test="tei:monogr/tei:imprint/tei:biblScope[@unit='page']/text()">
          <fpage>
            <xsl:value-of select="tei:monogr/tei:imprint/tei:biblScope[@unit='page']"/>
          </fpage>
          <lpage>
            <xsl:value-of select="tei:monogr/tei:imprint/tei:biblScope[@unit='page']"/>
          </lpage>
        </xsl:when>
      </xsl:choose>

      <xsl:for-each select=".//tei:ref[@type='url'][normalize-space()]">
        <xsl:call-template name="link">
          <xsl:with-param name="text" select="normalize-space()"/>
          <xsl:with-param name="text-element" select="'comment'"/>
        </xsl:call-template>
      </xsl:for-each>

      <xsl:variable name="doi" select="(tei:analytic | tei:monogr)/tei:idno[@type='DOI']"/>
      <xsl:if test="$doi">
        <pub-id pub-id-type="doi">
          <xsl:value-of select="$doi"/>
        </pub-id>
      </xsl:if>
    </element-citation>
  </xsl:template>

  <!-- letters and digits: a part is only found where it is not inside a longer word -->
  <xsl:variable name="word-characters"
    select="'0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZÀÁÂÃÄÅÇÈÉÊËÌÍÎÏÑÒÓÔÕÖØÙÚÛÜÝàáâãäåçèéêëìíîïñòóôõöøùúûüýÿ'"/>

  <!-- the printed reference, its parsed parts tagged where they are printed -->
  <xsl:template name="mixed-citation">
    <xsl:variable name="parts" select="
      tei:analytic/tei:author/tei:persName/tei:surname
      | tei:analytic/tei:author/tei:persName/tei:forename
      | tei:monogr/tei:author/tei:persName/tei:surname
      | tei:monogr/tei:author/tei:persName/tei:forename
      | tei:analytic/tei:title[@type='main']
      | tei:monogr/tei:title[@type='main'][not(@level='j')][not(../../tei:analytic/tei:title[@type='main'])]
      | tei:monogr/tei:title[@level='j']
      | tei:monogr/tei:imprint/tei:date[@type='published']
      | tei:monogr/tei:imprint/tei:biblScope[@unit='volume' or @unit='issue']
      | tei:monogr/tei:imprint/tei:biblScope[@unit='page'][not(@from or @to)]
      | tei:monogr/tei:imprint/tei:biblScope[@unit='page']/@from
      | tei:monogr/tei:imprint/tei:biblScope[@unit='page']/@to
    "/>
    <xsl:variable name="printed" select="string(tei:note[@type='raw_reference'])"/>
    <!-- A line break can leave a space inside a printed DOI or URL, so they are found
         ignoring whitespace, and written without it. -->
    <xsl:variable name="identifiers">
      <xsl:for-each select="tei:analytic/tei:idno[@type='DOI'] | tei:monogr/tei:idno[@type='DOI']
        | .//tei:ref[@type='url']">
        <xsl:variable name="value" select="normalize-space()"/>
        <xsl:variable name="printed-value">
          <xsl:call-template name="find-ignoring-whitespace">
            <xsl:with-param name="text" select="$printed"/>
            <xsl:with-param name="value" select="$value"/>
          </xsl:call-template>
        </xsl:variable>
        <xsl:if test="string($printed-value)">
          <cite:identifier kind="{local-name()}" value="{$value}"><xsl:value-of select="$printed-value"/></cite:identifier>
        </xsl:if>
      </xsl:for-each>
    </xsl:variable>
    <!-- initials are often printed run together, "MJ" for M and J -->
    <xsl:variable name="joined-forenames">
      <xsl:for-each select="(tei:analytic | tei:monogr)/tei:author/tei:persName[count(tei:forename) &gt; 1]">
        <cite:forenames author="{generate-id(ancestor::tei:author[1])}">
          <xsl:attribute name="covers">
            <xsl:for-each select="tei:forename">
              <xsl:value-of select="concat(generate-id(), '|')"/>
            </xsl:for-each>
          </xsl:attribute>
          <xsl:for-each select="tei:forename">
            <xsl:value-of select="normalize-space()"/>
          </xsl:for-each>
        </cite:forenames>
      </xsl:for-each>
    </xsl:variable>
    <xsl:variable name="tagged">
      <xsl:call-template name="cite-mixed-content">
        <xsl:with-param name="rest" select="$printed"/>
        <xsl:with-param name="parts" select="$parts | exsl:node-set($identifiers)/cite:identifier
          | exsl:node-set($joined-forenames)/cite:forenames"/>
      </xsl:call-template>
    </xsl:variable>
    <xsl:variable name="named">
      <xsl:apply-templates select="exsl:node-set($tagged)/node()" mode="cite-names"/>
    </xsl:variable>
    <mixed-citation publication-type="journal">
      <xsl:call-template name="cite-person-group">
        <xsl:with-param name="nodes" select="exsl:node-set($named)/node()"/>
      </xsl:call-template>
    </mixed-citation>
  </xsl:template>

  <!-- as the affiliation is, with each part only found as a whole word -->
  <xsl:template name="cite-mixed-content">
    <xsl:param name="rest"/>
    <xsl:param name="parts"/>
    <xsl:param name="placed" select="'|'"/>
    <xsl:variable name="found">
      <xsl:for-each select="$parts[normalize-space(.)][not(contains($placed, concat('|', generate-id(), '|')))]">
        <xsl:variable name="offset">
          <xsl:call-template name="word-offset">
            <xsl:with-param name="text" select="$rest"/>
            <xsl:with-param name="value" select="string(.)"/>
          </xsl:call-template>
        </xsl:variable>
        <xsl:if test="$offset &gt;= 0">
          <cite:found id="{generate-id()}" offset="{$offset}" length="{string-length(.)}"/>
        </xsl:if>
      </xsl:for-each>
    </xsl:variable>
    <xsl:variable name="first-found">
      <xsl:for-each select="exsl:node-set($found)/cite:found">
        <xsl:sort select="@offset" data-type="number"/>
        <xsl:sort select="@length" data-type="number" order="descending"/>
        <xsl:if test="position() = 1">
          <xsl:copy-of select="."/>
        </xsl:if>
      </xsl:for-each>
    </xsl:variable>
    <xsl:variable name="first" select="exsl:node-set($first-found)/cite:found"/>
    <xsl:choose>
      <xsl:when test="$first">
        <xsl:variable name="part" select="$parts[generate-id() = $first/@id]"/>
        <xsl:value-of select="substring($rest, 1, $first/@offset)"/>
        <xsl:apply-templates select="$part" mode="cite-part"/>
        <xsl:call-template name="cite-mixed-content">
          <xsl:with-param name="rest" select="substring($rest, $first/@offset + $first/@length + 1)"/>
          <xsl:with-param name="parts" select="$parts"/>
          <xsl:with-param name="placed" select="concat($placed, $first/@id, '|', $part/@covers)"/>
        </xsl:call-template>
      </xsl:when>
      <xsl:otherwise>
        <xsl:value-of select="$rest"/>
      </xsl:otherwise>
    </xsl:choose>
  </xsl:template>

  <!-- where value first stands in text as a whole word, as the number of characters
       before it; -1 where it does not -->
  <xsl:template name="word-offset">
    <xsl:param name="text"/>
    <xsl:param name="value"/>
    <xsl:param name="skipped" select="0"/>
    <xsl:param name="previous" select="''"/>
    <xsl:variable name="before" select="substring-before($text, $value)"/>
    <xsl:variable name="character-before" select="concat(
      substring($previous, 1, number($before = '')),
      substring($before, string-length($before), 1)
    )"/>
    <xsl:variable name="character-after" select="substring(substring-after($text, $value), 1, 1)"/>
    <xsl:choose>
      <xsl:when test="not(contains($text, $value))">-1</xsl:when>
      <xsl:when test="
        ($character-before = '' or not(contains($word-characters, $character-before)))
        and ($character-after = '' or not(contains($word-characters, $character-after)))
      ">
        <xsl:value-of select="$skipped + string-length($before)"/>
      </xsl:when>
      <xsl:otherwise>
        <xsl:call-template name="word-offset">
          <xsl:with-param name="text" select="substring($text, string-length($before) + 2)"/>
          <xsl:with-param name="value" select="$value"/>
          <xsl:with-param name="skipped" select="$skipped + string-length($before) + 1"/>
          <xsl:with-param name="previous" select="substring($text, string-length($before) + 1, 1)"/>
        </xsl:call-template>
      </xsl:otherwise>
    </xsl:choose>
  </xsl:template>

  <xsl:template match="cite:forenames" mode="cite-part">
    <given-names cite:author="{@author}"><xsl:value-of select="."/></given-names>
  </xsl:template>

  <xsl:template match="tei:surname" mode="cite-part">
    <surname cite:author="{generate-id(ancestor::tei:author[1])}"><xsl:value-of select="."/></surname>
  </xsl:template>

  <xsl:template match="tei:forename" mode="cite-part">
    <given-names cite:author="{generate-id(ancestor::tei:author[1])}"><xsl:value-of select="."/></given-names>
  </xsl:template>

  <xsl:template match="tei:title[@level='j']" mode="cite-part">
    <source><xsl:value-of select="."/></source>
  </xsl:template>

  <xsl:template match="tei:title" mode="cite-part">
    <article-title><xsl:value-of select="."/></article-title>
  </xsl:template>

  <xsl:template match="tei:date" mode="cite-part">
    <year><xsl:value-of select="."/></year>
  </xsl:template>

  <xsl:template match="tei:biblScope[@unit='volume']" mode="cite-part">
    <volume><xsl:value-of select="."/></volume>
  </xsl:template>

  <xsl:template match="tei:biblScope[@unit='issue']" mode="cite-part">
    <issue><xsl:value-of select="."/></issue>
  </xsl:template>

  <xsl:template match="tei:biblScope | tei:biblScope/@from" mode="cite-part">
    <fpage><xsl:value-of select="."/></fpage>
  </xsl:template>

  <xsl:template match="tei:biblScope/@to" mode="cite-part">
    <lpage><xsl:value-of select="."/></lpage>
  </xsl:template>

  <xsl:template match="cite:identifier[@kind='idno']" mode="cite-part">
    <pub-id pub-id-type="doi"><xsl:value-of select="@value"/></pub-id>
  </xsl:template>

  <xsl:template match="cite:identifier[@kind='ref']" mode="cite-part">
    <xsl:call-template name="link">
      <xsl:with-param name="text" select="@value"/>
    </xsl:call-template>
  </xsl:template>

  <!-- an ext-link, or the text where the schema would reject its address, in text-element if given -->
  <xsl:template name="link">
    <xsl:param name="text"/>
    <xsl:param name="text-element"/>
    <xsl:variable name="url" select="translate($text, '&lt;&gt; ', '')"/>
    <xsl:variable name="host-and-port">
      <xsl:call-template name="host-and-port">
        <xsl:with-param name="url" select="$url"/>
      </xsl:call-template>
    </xsl:variable>
    <xsl:variable name="port" select="substring-after($host-and-port, ':')"/>
    <xsl:variable name="escapes-are-whole">
      <xsl:call-template name="escapes-are-whole">
        <xsl:with-param name="text" select="$url"/>
      </xsl:call-template>
    </xsl:variable>
    <xsl:choose>
      <xsl:when test="
        (not(contains($host-and-port, ':')) or ($port != '' and translate($port, '0123456789', '') = ''))
        and not(contains(substring-after($url, '#'), '#'))
        and $escapes-are-whole = 'true'
      ">
        <ext-link ext-link-type="uri" xlink:href="{$url}"><xsl:value-of select="$text"/></ext-link>
      </xsl:when>
      <xsl:when test="$text-element">
        <xsl:element name="{$text-element}"><xsl:value-of select="$text"/></xsl:element>
      </xsl:when>
      <xsl:otherwise>
        <xsl:value-of select="$text"/>
      </xsl:otherwise>
    </xsl:choose>
  </xsl:template>

  <xsl:template name="host-and-port">
    <xsl:param name="url"/>
    <xsl:if test="contains($url, '://')">
      <xsl:variable name="authority" select="substring-before(
        translate(concat(substring-after($url, '://'), '/'), '?#', '//'), '/'
      )"/>
      <xsl:variable name="host-and-port">
        <xsl:choose>
          <xsl:when test="contains($authority, '@')">
            <xsl:value-of select="substring-after($authority, '@')"/>
          </xsl:when>
          <xsl:otherwise><xsl:value-of select="$authority"/></xsl:otherwise>
        </xsl:choose>
      </xsl:variable>
      <xsl:choose>
        <xsl:when test="starts-with($host-and-port, '[')">
          <xsl:value-of select="substring-after($host-and-port, ']')"/>
        </xsl:when>
        <xsl:otherwise><xsl:value-of select="$host-and-port"/></xsl:otherwise>
      </xsl:choose>
    </xsl:if>
  </xsl:template>

  <!-- 'true' where every % starts a two-digit hex escape -->
  <xsl:template name="escapes-are-whole">
    <xsl:param name="text"/>
    <xsl:choose>
      <xsl:when test="not(contains($text, '%'))">true</xsl:when>
      <xsl:when test="
        string-length(substring-after($text, '%')) &gt;= 2
        and translate(substring(substring-after($text, '%'), 1, 2), '0123456789abcdefABCDEF', '') = ''
      ">
        <xsl:call-template name="escapes-are-whole">
          <xsl:with-param name="text" select="substring(substring-after($text, '%'), 3)"/>
        </xsl:call-template>
      </xsl:when>
      <xsl:otherwise>false</xsl:otherwise>
    </xsl:choose>
  </xsl:template>

  <!-- where value is printed in text, with any whitespace inside it; empty if it is not -->
  <xsl:template name="find-ignoring-whitespace">
    <xsl:param name="text"/>
    <xsl:param name="value"/>
    <xsl:variable name="compact-text" select="translate($text, $whitespace, '')"/>
    <xsl:variable name="compact-value" select="translate($value, $whitespace, '')"/>
    <xsl:if test="$compact-value and contains($compact-text, $compact-value)">
      <xsl:variable name="before-length">
        <xsl:call-template name="non-whitespace-prefix-length">
          <xsl:with-param name="text" select="$text"/>
          <xsl:with-param name="count"
            select="string-length(substring-before($compact-text, $compact-value)) + 1"/>
        </xsl:call-template>
      </xsl:variable>
      <xsl:variable name="start" select="substring($text, $before-length)"/>
      <xsl:variable name="length">
        <xsl:call-template name="non-whitespace-prefix-length">
          <xsl:with-param name="text" select="$start"/>
          <xsl:with-param name="count" select="string-length($compact-value)"/>
        </xsl:call-template>
      </xsl:variable>
      <xsl:value-of select="substring($start, 1, $length)"/>
    </xsl:if>
  </xsl:template>

  <xsl:variable name="whitespace" select="'&#9;&#10;&#13; &#160;'"/>

  <!-- the length of the shortest start of text holding count characters that are not
       whitespace, halving the text each step so the recursion stays shallow -->
  <xsl:template name="non-whitespace-prefix-length">
    <xsl:param name="text"/>
    <xsl:param name="count"/>
    <xsl:variable name="half" select="floor(string-length($text) div 2)"/>
    <xsl:variable name="first-count"
      select="string-length(translate(substring($text, 1, $half), $whitespace, ''))"/>
    <xsl:choose>
      <xsl:when test="$count &lt;= 0">0</xsl:when>
      <xsl:when test="string-length($text) &lt;= 1">
        <xsl:value-of select="string-length($text)"/>
      </xsl:when>
      <xsl:when test="$first-count &gt;= $count">
        <xsl:call-template name="non-whitespace-prefix-length">
          <xsl:with-param name="text" select="substring($text, 1, $half)"/>
          <xsl:with-param name="count" select="$count"/>
        </xsl:call-template>
      </xsl:when>
      <xsl:otherwise>
        <xsl:variable name="rest-length">
          <xsl:call-template name="non-whitespace-prefix-length">
            <xsl:with-param name="text" select="substring($text, $half + 1)"/>
            <xsl:with-param name="count" select="$count - $first-count"/>
          </xsl:call-template>
        </xsl:variable>
        <xsl:value-of select="$half + $rest-length"/>
      </xsl:otherwise>
    </xsl:choose>
  </xsl:template>

  <!-- An author's name parts become one string-name where they are printed together,
       with only spaces and punctuation between them; otherwise they stay plain text. -->
  <xsl:template match="text()" mode="cite-names">
    <xsl:variable name="author" select="preceding-sibling::node()[1]/@cite:author"/>
    <xsl:variable name="is-inside-a-name" select="
      translate(., ' ,.&#160;', '') = ''
      and $author
      and following-sibling::node()[1]/@cite:author = $author
      and count(../*[@cite:author = $author][
        not(preceding-sibling::node()[1]/@cite:author = @cite:author)
        and not(
          preceding-sibling::node()[1][self::text()][translate(., ' ,.&#160;', '') = '']
          and preceding-sibling::node()[2]/@cite:author = @cite:author
        )
      ]) = 1
    "/>
    <xsl:if test="not($is-inside-a-name)">
      <xsl:value-of select="."/>
    </xsl:if>
  </xsl:template>

  <xsl:template match="*[@cite:author]" mode="cite-names">
    <xsl:variable name="author" select="@cite:author"/>
    <xsl:variable name="name-starts" select="../*[@cite:author = $author][
      not(preceding-sibling::node()[1]/@cite:author = @cite:author)
      and not(
        preceding-sibling::node()[1][self::text()][translate(., ' ,.&#160;', '') = '']
        and preceding-sibling::node()[2]/@cite:author = @cite:author
      )
    ]"/>
    <xsl:choose>
      <xsl:when test="count($name-starts) != 1">
        <xsl:value-of select="."/>
      </xsl:when>
      <xsl:when test="generate-id($name-starts) = generate-id(.)">
        <string-name>
          <xsl:for-each select=". | following-sibling::node()[
            @cite:author = $author
            or (
              self::text()
              and preceding-sibling::node()[1]/@cite:author = $author
              and following-sibling::node()[1]/@cite:author = $author
            )
          ]">
            <xsl:choose>
              <xsl:when test="self::*">
                <xsl:element name="{local-name()}"><xsl:value-of select="."/></xsl:element>
              </xsl:when>
              <xsl:otherwise>
                <xsl:value-of select="."/>
              </xsl:otherwise>
            </xsl:choose>
          </xsl:for-each>
        </string-name>
      </xsl:when>
    </xsl:choose>
  </xsl:template>

  <xsl:template match="*" mode="cite-names">
    <xsl:copy-of select="."/>
  </xsl:template>

  <!-- the names, from the first to the last, as one person-group where nothing else
       is tagged between them -->
  <xsl:template name="cite-person-group">
    <xsl:param name="nodes"/>
    <xsl:variable name="names" select="$nodes[self::string-name]"/>
    <xsl:variable name="between" select="$nodes[
      preceding-sibling::string-name and following-sibling::string-name
    ]"/>
    <xsl:choose>
      <xsl:when test="$names and not($between[self::*][not(self::string-name)])">
        <xsl:copy-of select="$nodes[not(self::string-name) and not(preceding-sibling::string-name)]"/>
        <person-group person-group-type="author">
          <xsl:copy-of select="$names | $between"/>
        </person-group>
        <xsl:copy-of select="$nodes[not(self::string-name) and not(following-sibling::string-name)
          and preceding-sibling::string-name]"/>
      </xsl:when>
      <xsl:otherwise>
        <xsl:copy-of select="$nodes"/>
      </xsl:otherwise>
    </xsl:choose>
  </xsl:template>


  <xsl:template match="tei:persName">
    <name>
      <surname>
        <xsl:value-of select="tei:surname"/>
      </surname>
      <given-names>
        <xsl:for-each select="tei:forename">
          <xsl:if test="position() > 1">
            <xsl:value-of select="' '"/>
          </xsl:if>
          <xsl:value-of select="string(.)"/>
        </xsl:for-each>
      </given-names>
    </name>
  </xsl:template>

  <xsl:template match="tei:ref">
    <xsl:choose>
      <xsl:when test="@type = 'bibr' and @target">
        <xref ref-type="bibr">
          <xsl:attribute name='rid'>
            <xsl:value-of select="substring-after(@target, '#')"/>
          </xsl:attribute>
          <xsl:value-of select="."/>
        </xref>
      </xsl:when>
      <xsl:when test="@type = 'figure' and @target">
        <xref ref-type="fig">
          <xsl:attribute name='rid'>
            <xsl:value-of select="substring-after(@target, '#')"/>
          </xsl:attribute>
          <xsl:value-of select="."/>
        </xref>
      </xsl:when>
      <xsl:when test="@type = 'table' and @target">
        <xref ref-type="table">
          <xsl:attribute name='rid'>
            <xsl:value-of select="substring-after(@target, '#')"/>
          </xsl:attribute>
          <xsl:value-of select="."/>
        </xref>
      </xsl:when>
      <xsl:otherwise>
        <xsl:value-of select="."/>
      </xsl:otherwise>
    </xsl:choose>
  </xsl:template>

  <xsl:template match="tei:head">
    <title><xsl:apply-templates select="node()"/></title>
  </xsl:template>

  <xsl:template match="tei:title">
    <xsl:apply-templates select="node()"/>
  </xsl:template>

  <xsl:template match="tei:p">
    <p>
      <xsl:apply-templates select="node()"/>
    </p>
  </xsl:template>

  <xsl:template match="tei:hi[@rend='italic']">
    <xsl:choose>
      <xsl:when test="$output_italic = 'true'">
        <italic><xsl:apply-templates select="node()"/></italic>
      </xsl:when>
      <xsl:otherwise>
        <xsl:apply-templates select="node()"/>
      </xsl:otherwise>
    </xsl:choose>
  </xsl:template>

  <xsl:template match="tei:hi[@rend='bold']">
    <xsl:choose>
      <xsl:when test="$output_bold = 'true'">
        <bold><xsl:apply-templates select="node()"/></bold>
      </xsl:when>
      <xsl:otherwise>
        <xsl:apply-templates select="node()"/>
      </xsl:otherwise>
    </xsl:choose>
  </xsl:template>

  <!--
  Template: parseDateComponents (XSLT 1.0)

  Params:
    date: ISO year, year-month or year-month-day
  
  Examples:
    date=2001 => <year>2001</year>
    date=2001-02 => <year>2001</year><month>02</month>
    date=2001-02-03 => <year>2001</year><month>02</month><day>03</day>
  -->
  <xsl:template name="parseDateComponents">
    <xsl:param name="date" select="."/>
    <xsl:call-template name="_parseDateComponentYearMonthDay">
      <xsl:with-param name="value" select="$date"/>
    </xsl:call-template>
  </xsl:template>

  <xsl:template name="_parseDateComponentYearMonthDay">
    <xsl:param name="value" select="."/>
    <xsl:choose>
      <xsl:when test="contains($value, '-')">
        <year>
          <xsl:value-of select="substring-before($value, '-')"/>
        </year>
        <xsl:call-template name="_parseDateComponentMonthDay">
          <xsl:with-param name="value" select="substring-after($value, '-')"/>
        </xsl:call-template>
      </xsl:when>
      <xsl:otherwise>
        <year>
          <xsl:value-of select="$value"/>
        </year>
      </xsl:otherwise>
    </xsl:choose>
  </xsl:template>

  <xsl:template name="_parseDateComponentMonthDay">
    <xsl:param name="value" select="."/>
    <xsl:choose>
      <xsl:when test="contains($value, '-')">
        <month>
          <xsl:value-of select="substring-before($value, '-')"/>
        </month>
        <day>
          <xsl:value-of select="substring-after($value, '-')"/>
        </day>
      </xsl:when>
      <xsl:otherwise>
        <month>
          <xsl:value-of select="$value"/>
        </month>
      </xsl:otherwise>
    </xsl:choose>
  </xsl:template>

</xsl:stylesheet>