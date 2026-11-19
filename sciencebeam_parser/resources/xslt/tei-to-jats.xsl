<?xml version="1.0" encoding="UTF-8"?>
<xsl:stylesheet 
  xmlns:xsl="http://www.w3.org/1999/XSL/Transform" 
  xmlns:xs="http://www.w3.org/2001/XMLSchema" 
  xmlns:mml="http://www.w3.org/1998/Math/MathML" 
  xmlns:xlink="http://www.w3.org/1999/xlink" 
  xmlns:tei="http://www.tei-c.org/ns/1.0"
  exclude-result-prefixes="xs mml tei"
  version="1.0"
>
  <xsl:param name="output_parameters" select="'false'"/>
  <xsl:param name="output_bold" select="'false'"/>
  <xsl:param name="output_italic" select="'false'"/>
  <xsl:param name="output_empty_figure_graphic" select="'true'"/>
  <xsl:param name="acknowledgement_target" select="'ack'"/>
  <xsl:param name="annex_target" select="'back'"/>

  <xsl:template match="/">
    <article article-type="research-article">
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
      <title-group>
        <article-title>
          <xsl:copy-of select="tei:fileDesc/tei:titleStmt/tei:title[1]/@xml:lang"/>
          <xsl:apply-templates select="tei:fileDesc/tei:titleStmt/tei:title[1]"/>
        </article-title>
        <xsl:for-each select="tei:fileDesc/tei:titleStmt/tei:title[position() &gt; 1]">
          <trans-title-group>
            <xsl:copy-of select="@xml:lang"/>
            <trans-title>
              <xsl:apply-templates select="."/>
            </trans-title>
          </trans-title-group>
        </xsl:for-each>
      </title-group>

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
          <xsl:copy-of select="@xml:lang"/>
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
    </article-meta>
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

  <xsl:template match="tei:graphic">
    <graphic>
      <xsl:if test="@url">
        <xsl:attribute name='xlink:href'>
          <xsl:value-of select="@url"/>
        </xsl:attribute>
      </xsl:if>
    </graphic>
  </xsl:template>

  <xsl:template match="tei:figure[not(@type='table')]">
    <fig>
      <xsl:attribute name='id'>
        <xsl:value-of select="@xml:id"/>
      </xsl:attribute>
      <object-id><xsl:value-of select="@xml:id"/></object-id>
      <label><xsl:value-of select="tei:head"/></label>
      <caption>
        <xsl:apply-templates select="tei:head"/>
        <p><xsl:apply-templates select="tei:figDesc"/></p>
      </caption>
      <xsl:apply-templates select="tei:graphic"/>
      <xsl:if test="$output_empty_figure_graphic = 'true'">
        <xsl:if test="not(tei:graphic)">
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
      <label><xsl:value-of select="tei:head"/></label>
      <caption>
        <xsl:apply-templates select="tei:head"/>
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
            <xsl:apply-templates select="tei:div[@type='annex']/tei:div"/>
            <xsl:apply-templates select="tei:div[@type='annex']/tei:figure"/>
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

        <xsl:variable name="doi" select="(tei:analytic | tei:monogr)/tei:idno[@type='DOI']"/>
        <xsl:if test="$doi">
          <pub-id pub-id-type="doi">
            <xsl:value-of select="$doi"/>
          </pub-id>
        </xsl:if>
      </element-citation>
    </ref>
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
        <i><xsl:apply-templates select="node()"/></i>
      </xsl:when>
      <xsl:otherwise>
        <xsl:apply-templates select="node()"/>
      </xsl:otherwise>
    </xsl:choose>
  </xsl:template>

  <xsl:template match="tei:hi[@rend='bold']">
    <xsl:choose>
      <xsl:when test="$output_bold = 'true'">
        <b><xsl:apply-templates select="node()"/></b>
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