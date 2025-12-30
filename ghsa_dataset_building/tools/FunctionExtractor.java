import com.github.javaparser.*;
import com.github.javaparser.ast.body.*;
import com.github.javaparser.ast.*;

import java.io.File;
import java.util.Optional;

public class FunctionExtractor {
    public static void main(String[] args) throws Exception {
        File file = new File(args[0]);
        int lineNumber = Integer.parseInt(args[1]);

        CompilationUnit cu = StaticJavaParser.parse(file);

        cu.findAll(MethodDeclaration.class).forEach(m -> {
            if (m.getRange().isPresent()) {
                Range r = m.getRange().get();
                if (r.begin.line <= lineNumber && lineNumber <= r.end.line) {
                    System.out.println(m.toString());
                }
            }
        });

        cu.findAll(ClassOrInterfaceDeclaration.class).forEach(c -> {
            if (c.getRange().isPresent()) {
                Range r = c.getRange().get();
                if (r.begin.line <= lineNumber && lineNumber <= r.end.line) {
                    System.out.println(c.toString());
                }
            }
        });
    }
}
